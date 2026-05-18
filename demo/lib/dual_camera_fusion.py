"""
Dual-camera court-space fusion (FL + NL) — the glue layer.

DESIGN NOTE (deviation from 06_DUAL_CAMERA_FUSION.md §6, found in testing):
trackingStudio's `CrossCameraMerger` only matches a detection against global
players carried from PRIOR frames — its `_associate_players_across_cameras`
merely logs candidate pairs, it never links them. So two cameras seeing the
same player for the first time in the same frame each spawn a separate global
player and never reconcile. The `SpatialCrossCameraMerger` threshold fix does
NOT cure this (structural, not a threshold). `demo/tests/test_dual_fusion.py`
encodes that finding. We therefore do our OWN per-frame court-space fusion
here — simple, correct, testable — and keep `cross_camera_merger.py` /
`spatial_merger.py` vendored as REFERENCE only (provenance + the analysis).

What this does each synchronized FL/NL frame-pair:
  1. Project every detection's FOOT point (bbox bottom-center) to shared court
     CM via that camera's homography (vendored CalibrationIntegration).
  2. Greedy nearest-neighbour match FL<->NL within `max_player_distance_cm`
     (mutual, closest-first) -> one fused dot at the mean position.
  3. Unmatched detections pass through as single-camera dots.
This dedups the overlap band while keeping single-camera-only players, which
is all the uniform-dot demo needs (persistent global IDs are unnecessary —
every dot is identical; optional trails can use the per-camera track_id).

Camera id convention: FL = 1, NL = 2.
Fisheye (NL): optional pluggable `*_undistort` callable; no-op by default
(06 §9 default = central-fit). Pure library: no S3/ffmpeg/GUI -> unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .calibration_integration import CalibrationIntegration

BBox = Tuple[float, float, float, float]          # x1, y1, x2, y2 (image px)
Warp = Callable[[Tuple[float, float]], Tuple[float, float]]

FL_CAM_ID = 1
NL_CAM_ID = 2


def foot_point(bbox: BBox) -> Tuple[float, float]:
    """Bottom-center of the bbox = where the player meets the floor."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


@dataclass
class RawDetection:
    """One player box from a single camera's per-frame tracker output."""
    track_id: int
    bbox: BBox
    confidence: float = 1.0


@dataclass
class FusedPlayer:
    """A player in shared court space (cm) for one frame."""
    court_xy: Tuple[float, float]
    cameras: List[int] = field(default_factory=list)
    # per-camera source track ids (for optional trails / debugging)
    track_ids: dict = field(default_factory=dict)


def _project(calib: CalibrationIntegration, bbox: BBox,
             undistort: Optional[Warp]) -> Tuple[float, float]:
    fx, fy = foot_point(bbox)
    if undistort is not None:
        fx, fy = undistort((fx, fy))
    cx, cy = calib.image_to_court(fx, fy)   # uses inverse homography -> court cm
    return float(cx), float(cy)


def _greedy_match(fl_pts: List[Tuple[float, float]],
                  nl_pts: List[Tuple[float, float]],
                  max_dist: float) -> List[Tuple[int, int]]:
    """Mutual greedy nearest-neighbour, closest pair first."""
    cands: List[Tuple[float, int, int]] = []
    for i, (ax, ay) in enumerate(fl_pts):
        for j, (bx, by) in enumerate(nl_pts):
            d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if d < max_dist:
                cands.append((d, i, j))
    cands.sort()
    used_fl: set = set()
    used_nl: set = set()
    pairs: List[Tuple[int, int]] = []
    for _d, i, j in cands:
        if i in used_fl or j in used_nl:
            continue
        pairs.append((i, j))
        used_fl.add(i)
        used_nl.add(j)
    return pairs


class DualCameraFusion:
    """FL + NL detections -> unified players in court CM space (per frame)."""

    def __init__(self, fl_calibration_json: str, nl_calibration_json: str,
                 max_player_distance_cm: float = 150.0,
                 fl_undistort: Optional[Warp] = None,
                 nl_undistort: Optional[Warp] = None):
        self.fl = CalibrationIntegration(fl_calibration_json)
        self.nl = CalibrationIntegration(nl_calibration_json)
        for name, calib in (("FL", self.fl), ("NL", self.nl)):
            if calib.inverse_homography is None:
                raise ValueError(
                    f"{name} calibration has no homography — run Step 1 "
                    f"(demo/calibrate_homography.py) for that camera first")
        self.max_dist = float(max_player_distance_cm)
        self.fl_undistort = fl_undistort
        self.nl_undistort = nl_undistort

    def fuse(self, fl_dets: Sequence[RawDetection],
             nl_dets: Sequence[RawDetection]) -> List[FusedPlayer]:
        fl_pts = [_project(self.fl, d.bbox, self.fl_undistort) for d in fl_dets]
        nl_pts = [_project(self.nl, d.bbox, self.nl_undistort) for d in nl_dets]

        pairs = _greedy_match(fl_pts, nl_pts, self.max_dist)
        matched_fl = {i for i, _ in pairs}
        matched_nl = {j for _, j in pairs}
        fused: List[FusedPlayer] = []

        for i, j in pairs:                                  # overlap -> merged
            (ax, ay), (bx, by) = fl_pts[i], nl_pts[j]
            fused.append(FusedPlayer(
                court_xy=((ax + bx) / 2.0, (ay + by) / 2.0),
                cameras=[FL_CAM_ID, NL_CAM_ID],
                track_ids={FL_CAM_ID: fl_dets[i].track_id,
                           NL_CAM_ID: nl_dets[j].track_id},
            ))
        for i, p in enumerate(fl_pts):                      # FL-only
            if i not in matched_fl:
                fused.append(FusedPlayer(court_xy=p, cameras=[FL_CAM_ID],
                                         track_ids={FL_CAM_ID: fl_dets[i].track_id}))
        for j, p in enumerate(nl_pts):                      # NL-only
            if j not in matched_nl:
                fused.append(FusedPlayer(court_xy=p, cameras=[NL_CAM_ID],
                                         track_ids={NL_CAM_ID: nl_dets[j].track_id}))
        return fused
