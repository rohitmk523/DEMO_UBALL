"""
Unit tests for the dual-camera court-space fusion.

Fully synthetic (no S3 / ffmpeg / GUI) — the executable proof that the fusion
math is correct while real per-camera homographies still await the operator
click pass (Step 1, headless-blocked).

`test_vendored_merger_cannot_unify_same_frame` encodes the structural finding
documented in dual_camera_fusion.py's header / 06 §6: trackingStudio's merger
only matches against prior-frame globals, so it CANNOT merge two cameras seeing
a player for the first time in the same frame — which is why we do our own
per-frame fusion.

    pytest demo/tests/test_dual_fusion.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # demo/
from lib.cross_camera_merger import CrossCameraMerger  # noqa: E402
from lib.spatial_merger import SpatialCrossCameraMerger  # noqa: E402
from lib.dual_camera_fusion import (  # noqa: E402
    DualCameraFusion, RawDetection, foot_point, FL_CAM_ID, NL_CAM_ID,
)

IDENTITY_H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _write_identity_calibration(path: Path) -> str:
    """Calibration whose court<->image map is identity, so a foot pixel
    (px, py) projects to court (px, py) — predictable for assertions."""
    path.write_text(json.dumps({
        "homography_matrix": IDENTITY_H,
        "court_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "image_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }))
    return str(path)


def _person_dicts(cam_id: int, foot_xy, track_id=1):
    return [{
        "class_name": "person", "track_id": track_id,
        "player_id": f"C{cam_id}_T{track_id}",
        "bbox": [foot_xy[0] - 10, foot_xy[1] - 40, foot_xy[0] + 10, foot_xy[1]],
        "center": (float(foot_xy[0]), float(foot_xy[1])),
        "confidence": 0.9,
    }]


# --- pure helpers -----------------------------------------------------------

@pytest.mark.unit
def test_foot_point_is_bottom_center():
    assert foot_point((100.0, 200.0, 140.0, 360.0)) == (120.0, 360.0)


# --- documented limitation of the vendored merger ---------------------------

@pytest.mark.unit
def test_vendored_merger_cannot_unify_same_frame():
    """Both the vendored merger AND the spatial subclass fail to merge two
    cameras seeing one player in the SAME first frame — the limitation is
    structural (matches only prior-frame globals), not a threshold. This is
    why DualCameraFusion does its own per-frame fusion."""
    H = np.array(IDENTITY_H, dtype=np.float64)
    fl = _person_dicts(1, (1000.0, 500.0))
    nl = _person_dicts(2, (1005.0, 505.0))  # ~7 cm away => same player

    base = CrossCameraMerger(max_player_distance_bev=150.0)
    assert len(base.merge_camera_detections(fl, nl, 0, 0.0, H, H)) == 2

    spatial = SpatialCrossCameraMerger(max_player_distance_bev=150.0)
    assert len(spatial.merge_camera_detections(fl, nl, 0, 0.0, H, H)) == 2


# --- our per-frame fusion (the path the demo uses) --------------------------

@pytest.mark.integration
def test_dual_fusion_merges_overlap_and_keeps_distinct(tmp_path):
    fl_json = _write_identity_calibration(tmp_path / "fl.json")
    nl_json = _write_identity_calibration(tmp_path / "nl.json")
    fusion = DualCameraFusion(fl_json, nl_json, max_player_distance_cm=150.0)

    # identity calib => court == foot px == bbox bottom-center.
    # Player A: in BOTH cameras' overlap ~same spot -> 1 fused, 2 cams.
    # Player B: only FL, far away -> 1 fused, FL only.
    fl = [
        RawDetection(track_id=1, bbox=(990, 460, 1010, 500)),   # A via FL -> (1000,500)
        RawDetection(track_id=2, bbox=(40, 60, 60, 100)),       # B FL-only -> (50,100)
    ]
    nl = [
        RawDetection(track_id=9, bbox=(995, 465, 1015, 505)),   # A via NL -> (1005,505)
    ]
    fused = fusion.fuse(fl, nl)

    assert len(fused) == 2
    both = [p for p in fused if p.cameras == [FL_CAM_ID, NL_CAM_ID]]
    fl_only = [p for p in fused if p.cameras == [FL_CAM_ID]]
    assert len(both) == 1 and len(fl_only) == 1

    ax, ay = both[0].court_xy                 # mean of (1000,500) & (1005,505)
    assert ax == pytest.approx(1002.5) and ay == pytest.approx(502.5)
    assert both[0].track_ids == {FL_CAM_ID: 1, NL_CAM_ID: 9}

    bx, by = fl_only[0].court_xy
    assert bx == pytest.approx(50.0) and by == pytest.approx(100.0)


@pytest.mark.integration
def test_no_double_match_when_two_close_players(tmp_path):
    """Two players both in overlap: greedy must pair them 1-1, not cross-match."""
    fl_json = _write_identity_calibration(tmp_path / "fl.json")
    nl_json = _write_identity_calibration(tmp_path / "nl.json")
    fusion = DualCameraFusion(fl_json, nl_json, max_player_distance_cm=150.0)

    fl = [
        RawDetection(track_id=1, bbox=(90, 60, 110, 100)),      # -> (100,100)
        RawDetection(track_id=2, bbox=(290, 260, 310, 300)),    # -> (300,300)
    ]
    nl = [
        RawDetection(track_id=8, bbox=(295, 265, 315, 305)),    # -> (305,305) ~ p2
        RawDetection(track_id=9, bbox=(95, 65, 115, 105)),      # -> (105,105) ~ p1
    ]
    fused = fusion.fuse(fl, nl)
    assert len(fused) == 2
    assert all(p.cameras == [FL_CAM_ID, NL_CAM_ID] for p in fused)
    centers = sorted(round(p.court_xy[0]) for p in fused)
    assert centers == [102, 302]              # (100+105)/2, (300+305)/2


@pytest.mark.integration
def test_missing_homography_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"court_points": [], "image_points": []}))
    ok = _write_identity_calibration(tmp_path / "ok.json")
    with pytest.raises(ValueError, match="no homography"):
        DualCameraFusion(ok, str(bad))
