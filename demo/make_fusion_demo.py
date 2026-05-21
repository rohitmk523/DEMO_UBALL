#!/usr/bin/env python3
"""
Investor demo: 4-camera fusion player tracking.

LEFT  : 2x2 grid of the 4 live camera feeds (FL, FR, NL, NR).
RIGHT : one top-down court map (lib/court.draw_topdown_court) with ONE
        smoothed red dot per real player + a short fading trail.

Pipeline per synchronized 4-cam frame:
  1. Each camera runs its OWN ultralytics tracker (separate track-id
     state) at imgsz=1280 so distant/right-side players are not missed.
  2. Foot point = bbox bottom-center -> court cm via that cam's inv(H).
  3. Region-aware trust: drop detections off-court or outside the
     court-X band that camera "owns"; assign a per-camera, per-zone
     confidence WEIGHT (how much that camera is trusted at that X).
  4. Cross-camera merge: greedy mutual NN cluster within `merge_cm` so a
     player seen by 2+ cameras becomes ONE fused point.  The fused
     position is a CONFIDENCE-WEIGHTED mean (the camera whose accurate
     zone contains the point dominates), not a plain mean.
  5. Temporal smoothing: persistent fused tracks, EMA position, debounced
     birth, delayed death. Only smoothed tracks are drawn (kills flicker).
  6. Team colour: per torso-crop HSV jersey descriptor accumulated per
     fused track, clip-level k-means(2) -> Team A / Team B / Ref, locked
     with hysteresis so labels never flip per frame.

Frames are deinterlaced (field=top,scale=1920:1080) so pixel space
matches the homographies, exactly as the calibration frames were.

  python demo/make_fusion_demo.py --t0 1400 --dur 240 --fps 24
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib import court  # noqa: E402
from lib import team_color  # noqa: E402

CAL = Path("demo/calibration")
SEGCACHE = Path("demo/_segcache")   # persistent pulled-segment reuse cache
S3_BASE = (
    "s3://uball-videos-production/court-a/2026-03-19/"
    "c2a354fe-eb34-4980-af00/"
    "2026-03-19_c2a354fe-eb34-4980-af00"
)
ANGLES: Tuple[str, ...] = ("FL", "FR", "NL", "NR")
SCALE, PAD = 0.55, 55
OUT_H = 720                       # side-by-side panel height
CELL_W, CELL_H = 640, 360         # each grid cell

XY = Tuple[float, float]


# --------------------------------------------------------------------------- #
#  S3 segment pull
# --------------------------------------------------------------------------- #
def presign(s3_uri: str, expires: int = 3600) -> str:
    out = subprocess.run(
        ["aws", "s3", "presign", s3_uri, "--expires-in", str(expires)],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def pull_segment(angle: str, t0: float, dur: float, fps: int,
                 cache: bool) -> Path:
    """Pull a deinterlaced segment for one angle (HTTP range, no full DL).

    When ``cache`` is set the segment is written to / reused from the
    persistent ``demo/_segcache`` dir so repeated tuning runs (same
    t0/dur/fps) skip the S3 + ffmpeg round-trip entirely.
    """
    SEGCACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{angle}_{int(t0)}_{int(dur)}_{fps}"
    out = SEGCACHE / (f"{tag}.mp4" if cache else f"_tmp_{tag}.mp4")
    if cache and out.exists() and out.stat().st_size > 0:
        print(f"  cache hit  {out.name}")
        return out
    url = presign(f"{S3_BASE}_{angle}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0),
         "-i", url, "-t", str(dur),
         "-vf", f"field=top,scale=1920:1080,fps={fps}",
         "-an", "-q:v", "3", str(out)],
        check=True,
    )
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"ffmpeg produced no segment for {angle}")
    return out


# --------------------------------------------------------------------------- #
#  Cross-camera per-frame merge (confidence-weighted)
# --------------------------------------------------------------------------- #
@dataclass
class Det:
    """One per-camera court-space detection carrying its trust weight."""
    x: float
    y: float
    w: float                       # per-camera, per-zone confidence weight
    angle: str
    tid: int                       # this camera's bytetrack id
    box: Tuple[float, float, float, float]
    cls: int = -1                  # detector class id (for ref pass-through)


@dataclass
class Fused:
    """A merged cross-camera observation for this frame."""
    x: float
    y: float
    members: List[Det]


def merge_points(dets: List[Det], merge_cm: float) -> List[Fused]:
    """Greedy mutual-NN cluster; cluster -> confidence-WEIGHTED mean.

    The camera whose accurate zone contains the point carries the most
    weight, so a loose camera (FR on the right) cannot drag the fused
    dot off a player when an accurate camera (NR) also sees them.
    """
    n = len(dets)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.hypot(dets[i].x - dets[j].x,
                               dets[i].y - dets[j].y))
            if d < merge_cm:
                pairs.append((d, i, j))
    pairs.sort()
    for _d, i, j in pairs:
        parent[find(i)] = find(j)

    clusters: Dict[int, List[Det]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(dets[i])

    out: List[Fused] = []
    for c in clusters.values():
        wsum = sum(d.w for d in c) or 1.0
        fx = sum(d.x * d.w for d in c) / wsum
        fy = sum(d.y * d.w for d in c) / wsum
        out.append(Fused(float(fx), float(fy), c))
    return out


# --------------------------------------------------------------------------- #
#  Temporal smoothing
# --------------------------------------------------------------------------- #
@dataclass
class SmoothTrack:
    sid: int
    pos: XY
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    trail: List[XY] = field(default_factory=list)
    members: List[Det] = field(default_factory=list)   # this-frame members


class Smoother:
    """Persistent fused tracks: EMA position, debounced birth, lazy death.

    Each smoothed track keeps a stable ``sid`` so per-fused-track team
    colour can be accumulated across the whole clip.
    """

    def __init__(self, match_cm: float, ema_new: float,
                 birth_hits: int, death_misses: int, trail_len: int):
        self.match_cm = match_cm
        self.ema_new = ema_new
        self.birth_hits = birth_hits
        self.death_misses = death_misses
        self.trail_len = trail_len
        self.tracks: List[SmoothTrack] = []
        self._next = 0

    def update(self, fused: List[Fused]) -> List[SmoothTrack]:
        cands: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            for pi, p in enumerate(fused):
                d = float(np.hypot(tr.pos[0] - p.x, tr.pos[1] - p.y))
                if d < self.match_cm:
                    cands.append((d, ti, pi))
        cands.sort()
        used_t: set = set()
        used_p: set = set()
        for _d, ti, pi in cands:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            tr = self.tracks[ti]
            f = fused[pi]
            a = self.ema_new
            tr.pos = (tr.pos[0] * (1 - a) + f.x * a,
                      tr.pos[1] * (1 - a) + f.y * a)
            tr.hits += 1
            tr.misses = 0
            tr.members = f.members
            if tr.hits >= self.birth_hits:
                tr.confirmed = True
            tr.trail.append(tr.pos)
            tr.trail[:] = tr.trail[-self.trail_len:]

        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
                tr.members = []
        for pi, f in enumerate(fused):
            if pi not in used_p:
                self.tracks.append(SmoothTrack(
                    sid=self._next, pos=(f.x, f.y),
                    trail=[(f.x, f.y)], members=f.members))
                self._next += 1

        self.tracks = [t for t in self.tracks
                       if t.misses <= self.death_misses]
        return [t for t in self.tracks if t.confirmed and t.misses == 0]


# --------------------------------------------------------------------------- #
#  Per-camera detection -> court points
# --------------------------------------------------------------------------- #
@dataclass
class CamCfg:
    angle: str
    Hinv: np.ndarray
    band_lo: float          # trusted court-X lower bound (cm)
    band_hi: float          # trusted court-X upper bound (cm)
    acc_lo: float           # accurate-zone court-X lower bound (cm)
    acc_hi: float           # accurate-zone court-X upper bound (cm)
    base_w: float           # global per-camera trust multiplier
    offset: int             # per-cam integer-second sync nudge


def cam_weight(cam: CamCfg, cx: float, falloff: float) -> float:
    """Per-camera, per-zone confidence weight at court-X = cx.

    Full ``base_w`` inside the camera's accurate zone; decays smoothly
    with distance OUTSIDE it (weight ~ base_w / (1 + dist/falloff)) so a
    loose camera still contributes a little but is dominated by whichever
    camera owns that part of the floor.
    """
    if cam.acc_lo <= cx <= cam.acc_hi:
        d = 0.0
    elif cx < cam.acc_lo:
        d = cam.acc_lo - cx
    else:
        d = cx - cam.acc_hi
    return cam.base_w / (1.0 + d / max(1.0, falloff))


def court_points(boxes: np.ndarray, ids: np.ndarray,
                 clss: Optional[np.ndarray], cam: CamCfg,
                 min_box_h: float, L: float, W: float, margin: float,
                 roi_y: float, x_inset: float, falloff: float) -> List[Det]:
    out: List[Det] = []
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        if (y2 - y1) < min_box_h:
            continue
        if y2 < roi_y:                 # feet above this = bench/far-wall/fold
            continue
        foot = np.array([[[(x1 + x2) / 2.0, y2]]], np.float64)
        cx, cy = cv2.perspectiveTransform(foot, cam.Hinv)[0, 0]
        # X must be strictly INSIDE both baselines by `x_inset` cm: detections
        # near the image horizon fold collapse onto the baseline edge and
        # would otherwise appear as phantom dots pinned at X~0 / X~L.
        if not (x_inset <= cx <= L - x_inset):
            continue
        if not (-margin <= cy <= W + margin):
            continue
        if not (cam.band_lo <= cx <= cam.band_hi):
            continue
        tid = int(ids[k]) if ids is not None and k < len(ids) else -1
        cls = int(clss[k]) if clss is not None and k < len(clss) else -1
        out.append(Det(
            x=float(cx), y=float(cy),
            w=cam_weight(cam, float(cx), falloff),
            angle=cam.angle, tid=tid, cls=cls,
            box=(float(x1), float(y1), float(x2), float(y2))))
    return out


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def label(img: np.ndarray, txt: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA)


def build_grid(frames: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
    cells = []
    for ang in ANGLES:
        f = frames.get(ang)
        cell = (cv2.resize(f, (CELL_W, CELL_H)) if f is not None
                else np.full((CELL_H, CELL_W, 3), 20, np.uint8))
        label(cell, ang)
        cells.append(cell)
    top = np.hstack([cells[0], cells[1]])
    bot = np.hstack([cells[2], cells[3]])
    return np.vstack([top, bot])               # 1280 x 720


def draw_court(base: np.ndarray, tracks: List[SmoothTrack],
               team_of: Dict[int, str],
               clf: team_color.TeamClassifier) -> np.ndarray:
    canvas = base.copy()
    for tr in tracks:
        label = team_of.get(tr.sid, team_color.LABEL_REF)
        # Use the outline colour for the trail: a pure-black fill team
        # has a white outline, so its trail is a fading white streak
        # (visible on the dark canvas). White-fill team uses dark trail.
        col = clf.team_outline_bgr(label)
        tl = tr.trail
        for k in range(1, len(tl)):
            a = k / len(tl)
            p0 = court.cm_to_canvas(tl[k - 1][0], tl[k - 1][1], SCALE, PAD)
            p1 = court.cm_to_canvas(tl[k][0], tl[k][1], SCALE, PAD)
            cv2.line(canvas, p0, p1,
                     (int(col[0] * a), int(col[1] * a), int(col[2] * a)),
                     2, cv2.LINE_AA)
    for tr in tracks:
        label = team_of.get(tr.sid, team_color.LABEL_REF)
        col = clf.team_bgr(label)
        ring = clf.team_outline_bgr(label)
        px, py = court.cm_to_canvas(tr.pos[0], tr.pos[1], SCALE, PAD)
        cv2.circle(canvas, (px, py), 11, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 11, ring, 2, cv2.LINE_AA)
    return canvas


def draw_legend(panel: np.ndarray,
                clf: team_color.TeamClassifier) -> None:
    """Team-colour legend, drawn on the FINAL (already-resized) panel so
    text is never squashed by the court aspect resize."""
    items = [("Team A", clf.team_bgr(team_color.LABEL_A)),
             ("Team B", clf.team_bgr(team_color.LABEL_B)),
             ("Ref",    clf.team_bgr(team_color.LABEL_REF))]
    h = panel.shape[0]
    y = h - 14
    x = 12
    cv2.rectangle(panel, (x - 6, y - 20),
                  (x + 312, y + 10), (0, 0, 0), -1)
    for name, col in items:
        # outline contrast based on fill brightness
        ring = (255, 255, 255) if sum(col) < 380 else (0, 0, 0)
        cv2.circle(panel, (x + 7, y - 5), 7, col, -1, cv2.LINE_AA)
        cv2.circle(panel, (x + 7, y - 5), 7, ring, 2, cv2.LINE_AA)
        cv2.putText(panel, name, (x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
                    cv2.LINE_AA)
        x += 104


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t0", type=float, default=1400.0)
    ap.add_argument("--dur", type=float, default=240.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default="demo/fusion_demo_4cam.mp4")
    ap.add_argument("--weights", default="yolo11l.pt")
    ap.add_argument("--detector", choices=("yolo", "rfdetr"), default="yolo",
                    help="yolo = ultralytics .track(); rfdetr = RF-DETR "
                         "(basketball-finetuned) + supervision ByteTrack")
    ap.add_argument("--rfdetr-weights", default="demo/rfdetr_bball.pth",
                    help="RF-DETR checkpoint (.pth) when --detector rfdetr")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detector confidence (lowered for far-side recall)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="YOLO inference size; 1280 keeps distant/right "
                         "players (640 default downscale misses them)")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass demo/_segcache (force fresh S3 pull)")
    ap.add_argument("--min-box-h", type=float, default=55.0)
    ap.add_argument("--court-margin", type=float, default=15.0,
                    help="cm tolerance just outside the sidelines (Y)")
    ap.add_argument("--x-inset", type=float, default=45.0,
                    help="require feet at least this many cm INSIDE both "
                         "baselines (kills horizon-fold phantom edge dots)")
    ap.add_argument("--roi-top", type=float, default=0.30,
                    help="ignore detections whose feet are above this "
                         "fraction of frame height (bench/far wall/horizon "
                         "fold = the camera's least-accurate far zone)")
    ap.add_argument("--merge-cm", type=float, default=150.0)
    ap.add_argument("--match-cm", type=float, default=180.0)
    ap.add_argument("--ema-new", type=float, default=0.40)
    ap.add_argument("--birth-hits", type=int, default=3)
    ap.add_argument("--death-misses", type=int, default=12)
    ap.add_argument("--trail-len", type=int, default=12)
    ap.add_argument("--per-cam-offsets", default="0,0,0,0",
                    help="integer-second sync nudges FL,FR,NL,NR")
    # --- region-aware fusion trust ----------------------------------- #
    # Trusted court-X band fractions (of court length L): a detection is
    # dropped outside its camera's band.  Accurate-zone fractions: inside
    # them the camera gets full weight; outside it decays (cam_weight).
    # Derived empirically from a per-camera projection diagnostic on the
    # t1400 synced frames:
    #   FL  - front-left:  accurate LOW-X (own corner); garbage at neg X.
    #   NL  - near-left:   accurate LOW-X (left key); usable to mid court.
    #   NR  - near-right:  accurate HIGH-X (right key/hoop it sits behind)
    #                      -- this is the TRUSTWORTHY right-side camera.
    #   FR  - front-right: HIGH-X but loose (homography overshoots past
    #                      the baseline); low base weight everywhere.
    ap.add_argument("--fl-band", default="0.0,0.58")
    ap.add_argument("--nl-band", default="0.0,0.70")
    ap.add_argument("--fr-band", default="0.55,1.0")
    ap.add_argument("--nr-band", default="0.30,1.0")
    ap.add_argument("--fl-acc", default="0.0,0.45",
                    help="FL accurate court-X fraction band")
    ap.add_argument("--nl-acc", default="0.0,0.52")
    ap.add_argument("--fr-acc", default="0.80,1.0")
    ap.add_argument("--nr-acc", default="0.60,1.0")
    ap.add_argument("--fl-w", type=float, default=1.0)
    ap.add_argument("--nl-w", type=float, default=1.0)
    ap.add_argument("--fr-w", type=float, default=0.35,
                    help="FR global trust (loose homography -> low)")
    ap.add_argument("--nr-w", type=float, default=1.15,
                    help="NR global trust (accurate right-side cam)")
    ap.add_argument("--weight-falloff", type=float, default=350.0,
                    help="cm; weight ~ base/(1+dist_outside_acc/falloff)")
    # --- team-colour thresholds (HSV, OpenCV 0..179 hue) ------------- #
    ap.add_argument("--skin-hue-lo", type=int, default=3)
    ap.add_argument("--skin-hue-hi", type=int, default=22)
    ap.add_argument("--color-min-sat", type=int, default=55)
    ap.add_argument("--color-min-val", type=int, default=45)
    ap.add_argument("--ref-min-sat", type=int, default=50)
    ap.add_argument("--ref-hue-margin", type=float, default=22.0)
    ap.add_argument("--team-lock-samples", type=int, default=10)
    ap.add_argument("--team-mode", choices=("brightness", "siglip"),
                    default="brightness",
                    help="siglip = SigLIP image embeddings + k-means (robust "
                         "on dark-vs-white kit on a red floor; slower)")
    ap.add_argument("--detector-classes", default="0",
                    help="comma-separated detector class ids to keep. "
                         "yolo11l-COCO: '0' (person). Basketball finetune: "
                         "'3,4,5,6,7,8' (player variants + referee).")
    ap.add_argument("--ref-class", type=int, default=-1,
                    help="detector class id of referee (locked to LABEL_REF "
                         "without going through team clustering). 8 for the "
                         "basketball-player-detection-3 finetune; -1 = none.")
    a = ap.parse_args(argv)

    L, W = court.COURT_LENGTH_CM, court.COURT_WIDTH_CM
    offs = [int(x) for x in a.per_cam_offsets.split(",")]
    bands = {"FL": a.fl_band, "FR": a.fr_band,
             "NL": a.nl_band, "NR": a.nr_band}
    accs = {"FL": a.fl_acc, "FR": a.fr_acc,
            "NL": a.nl_acc, "NR": a.nr_acc}
    bws = {"FL": a.fl_w, "FR": a.fr_w, "NL": a.nl_w, "NR": a.nr_w}
    cams: Dict[str, CamCfg] = {}
    for i, ang in enumerate(ANGLES):
        lo_f, hi_f = (float(v) for v in bands[ang].split(","))
        alo_f, ahi_f = (float(v) for v in accs[ang].split(","))
        cams[ang] = CamCfg(
            angle=ang,
            Hinv=np.linalg.inv(np.load(CAL / f"{ang}_c2a_H.npy")),
            band_lo=lo_f * L, band_hi=hi_f * L,
            acc_lo=alo_f * L, acc_hi=ahi_f * L,
            base_w=bws[ang], offset=offs[i],
        )

    if a.team_mode == "siglip":
        clf = team_color.SigLIPTeamClassifier(
            lock_samples=a.team_lock_samples, device="mps",
        )
    else:
        clf = team_color.TeamClassifier(
            skin_hue_lo=a.skin_hue_lo, skin_hue_hi=a.skin_hue_hi,
            min_sat=a.color_min_sat, min_val=a.color_min_val,
            ref_min_sat=a.ref_min_sat, ref_hue_margin=a.ref_hue_margin,
            lock_samples=a.team_lock_samples,
        )
    keep_classes = [int(c) for c in a.detector_classes.split(",")]

    models = None
    rf_model = None
    trackers: Dict[str, object] = {}
    if a.detector == "rfdetr":
        import supervision as sv
        try:
            from rfdetr import RFDETRSmall as _RFD
        except Exception:
            from rfdetr import RFDETRBase as _RFD
        print(f"loading RF-DETR from {a.rfdetr_weights} ...")
        rf_model = _RFD(pretrain_weights=a.rfdetr_weights)
        trackers = {ang: sv.ByteTrack(frame_rate=a.fps) for ang in ANGLES}
    else:
        from ultralytics import YOLO
        models = {ang: YOLO(a.weights) for ang in ANGLES}

    use_cache = not a.no_cache
    caps: Dict[str, cv2.VideoCapture] = {}
    try:
        for ang in ANGLES:
            t0 = a.t0 + cams[ang].offset
            print(f"pulling {ang} segment t0={t0} dur={a.dur} ...")
            seg = pull_segment(ang, t0, a.dur, a.fps, use_cache)
            caps[ang] = cv2.VideoCapture(str(seg))

        base = court.draw_topdown_court(scale=SCALE, pad=PAD)
        ch, cw = base.shape[:2]
        grid_w = CELL_W * 2          # 1280
        panel_w = int(OUT_H * cw / ch)
        title_h = 40
        out_w, out_h = grid_w + panel_w, OUT_H + title_h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(a.out, fourcc, a.fps, (out_w, out_h))

        smoother = Smoother(a.match_cm, a.ema_new, a.birth_hits,
                            a.death_misses, a.trail_len)
        n = total = 0
        while True:
            frames: Dict[str, Optional[np.ndarray]] = {}
            ok_any = False
            pooled: List[Det] = []
            for ang in ANGLES:
                ok, frame = caps[ang].read()
                if not ok:
                    frames[ang] = None
                    continue
                ok_any = True
                frames[ang] = frame
                if a.detector == "rfdetr":
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    dets = rf_model.predict(rgb, threshold=a.conf)
                    if len(dets) and keep_classes:
                        dets = dets[np.isin(dets.class_id, keep_classes)]
                    dets = trackers[ang].update_with_detections(dets)
                    if len(dets) == 0:
                        continue
                    boxes = dets.xyxy
                    ids = dets.tracker_id
                    clss = dets.class_id
                else:
                    res = models[ang].track(
                        frame, persist=True, classes=keep_classes, conf=a.conf,
                        imgsz=a.imgsz, tracker="bytetrack.yaml",
                        device="mps", verbose=False,
                    )[0]
                    if res.boxes is None or res.boxes.xyxy is None:
                        continue
                    boxes = res.boxes.xyxy.cpu().numpy()
                    ids = (res.boxes.id.cpu().numpy()
                           if res.boxes.id is not None else None)
                    clss = (res.boxes.cls.cpu().numpy()
                            if res.boxes.cls is not None else None)
                roi_y = a.roi_top * frame.shape[0]
                pooled.extend(court_points(boxes, ids, clss, cams[ang],
                                           a.min_box_h, L, W,
                                           a.court_margin, roi_y,
                                           a.x_inset, a.weight_falloff))
            if not ok_any:
                break
            n += 1

            fused = merge_points(pooled, a.merge_cm)
            shown = smoother.update(fused)
            total += len(shown)

            # Accumulate jersey colour per fused track (clip-level
            # k-means decides the team; here we just feed crops). If the
            # detector explicitly classed the member as referee, lock
            # the fused track to LABEL_REF without going through teams.
            for tr in shown:
                if a.ref_class >= 0 and any(
                        d.cls == a.ref_class for d in tr.members):
                    if hasattr(clf, "mark_ref"):
                        clf.mark_ref(tr.sid)
                    continue
                for d in tr.members:
                    fr = frames.get(d.angle)
                    if fr is not None:
                        clf.observe(tr.sid, fr, d.box)
            team_of = {tr.sid: clf.label(tr.sid) for tr in shown}

            grid = build_grid(frames)
            court_img = cv2.resize(
                draw_court(base, shown, team_of, clf),
                (panel_w, OUT_H),
                interpolation=cv2.INTER_AREA)
            draw_legend(court_img, clf)
            body = np.hstack([grid, court_img])
            title = np.zeros((title_h, out_w, 3), np.uint8)
            cv2.putText(title,
                        f"UBALL  -  4-camera fusion court tracking"
                        f"   |   {len(shown)} players",
                        (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(np.vstack([title, body]))
            if n % 100 == 0:
                print(f"  {n} frames  ({len(shown)} players)")

        vw.release()
        avg = total / max(n, 1)
        print(f"done: {n} frames, avg {avg:.2f} players/frame -> {a.out}")
        ca, cb = (getattr(clf, "cen", None) or (None, None))
        if ca:
            print(f"team centroids (brightness)  A={tuple(round(v,1) for v in ca)} "
                  f"B={tuple(round(v,1) for v in cb)}")
    finally:
        for c in caps.values():
            c.release()
        # Only purge transient (non-cache) segments; the persistent
        # demo/_segcache is intentionally kept for fast re-runs.
        for f in SEGCACHE.glob("_tmp_*.mp4"):
            f.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
