#!/usr/bin/env python3
"""
Investor demo: SAM2 mask-propagation player tracking (single camera, FR).

Shows SAM2's persistent-ID quality vs ByteTrack: every player keeps the
SAME object id (== same colour / label) across the whole clip, surviving
occlusions, because SAM2 propagates a per-object mask through video memory
rather than re-associating boxes frame-to-frame.

LEFT  : the FR camera frame with each tracked player's mask outline +
        obj-id label.
RIGHT : the top-down CAD court (lib/court.draw_topdown_court) with one
        stable-coloured dot + obj-id per SAM2-tracked player and a short
        fading trail.

Pipeline:
  1. Pull a deinterlaced FR segment from S3 (field=top,scale=1920:1080,
     fps) -- EXACTLY the calibration pixel space -- and extract it to a
     directory of sequential JPEGs (the SAM2 video predictor needs a
     frame dir 00000.jpg, 00001.jpg, ...).
  2. Build the SAM2 video predictor (Meta API, sam2_hiera base+ ckpt) on
     MPS (CPU fallback).
  3. RF-DETR (basketball-finetuned) detects players on a SEED frame ->
     one SAM2 object per player box (add_new_points_or_box, obj_id=i).
     We seed on the frame that yields the most confident detections in a
     small window so the object set is as complete as possible.
  4. ONE propagate_in_video pass -> per frame, per obj_id, a mask. Foot
     point = mask bottom-centre -> inv(H) -> court cm; on-court filter.
     obj_id is the persistent track id (stable palette colour). SAM2's
     video memory holds each player's id through occlusions for the whole
     clip -- the property this demo exists to show.
  5. Render side-by-side, write mp4 (mp4v) at `fps`.

Design note -- why seed-only (no mid-clip re-detection):
SAM2's per-object cost on Apple MPS is steep and super-linear once
objects are conditioned at multiple frames (~1.5 s/frame for 7 objects,
~7 s/frame for 13, and 20 objects exhausts MPS memory). Adding new
objects at later conditioning frames also trips a Metal matmul dtype
assertion under load. Both make mid-clip re-detection infeasible on this
machine within a sane time budget, so we seed a single bounded object set
up front and run exactly one propagation pass. (For tracking players who
enter mid-clip, move this to a CUDA box -- see the final report.)

SAM2 on MPS stores memory features as bfloat16 while the rest of the
graph is fp32; running propagation under
``torch.autocast("mps", bfloat16)`` makes the graph consistently bf16
(avoids the dtype mismatch and is ~15% faster), so propagation is wrapped
in autocast.

  python demo/make_sam2_demo.py --t0 1400 --dur 60 --fps 12
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from lib import court  # noqa: E402

CAL = Path("demo/calibration")
S3_BASE = (
    "s3://uball-videos-production/court-a/2026-03-19/"
    "c2a354fe-eb34-4980-af00/"
    "2026-03-19_c2a354fe-eb34-4980-af00"
)
ANGLE = "FR"
FRAME_DIR = Path("/tmp/sam2_frames")
SCALE, PAD = 0.55, 55
OUT_H = 720                       # side-by-side panel height
PLAYER_CLASSES = (3, 4, 5, 6, 7)  # basketball-finetune player class ids
SAM2_CKPT = (
    "/Users/rohitkale/Cellstrat/GitHub_Repositories/"
    "Uball_tracking/models/sam2/sam2_hiera_base_plus.pt"
)
SAM2_CFG = "configs/sam2/sam2_hiera_b+.yaml"

XY = Tuple[float, float]

# 24-colour stable palette (BGR), indexed by obj_id -> persistent colour.
_PALETTE = [
    (66, 99, 235), (87, 201, 80), (38, 200, 250), (190, 120, 60),
    (180, 80, 200), (60, 180, 235), (235, 160, 60), (120, 220, 120),
    (235, 90, 160), (90, 235, 220), (200, 90, 90), (140, 200, 240),
    (60, 130, 235), (180, 235, 70), (235, 70, 110), (110, 90, 235),
    (70, 215, 175), (235, 200, 90), (160, 60, 200), (90, 200, 60),
    (235, 120, 200), (60, 235, 130), (200, 160, 235), (235, 235, 90),
]


def palette(obj_id: int) -> Tuple[int, int, int]:
    return _PALETTE[obj_id % len(_PALETTE)]


# --------------------------------------------------------------------------- #
#  S3 segment pull + frame extraction
# --------------------------------------------------------------------------- #
def presign(s3_uri: str, expires: int = 3600) -> str:
    out = subprocess.run(
        ["aws", "s3", "presign", s3_uri, "--expires-in", str(expires)],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def extract_frames(t0: float, dur: float, fps: int,
                   frame_dir: Path) -> List[Path]:
    """Pull a deinterlaced FR segment and explode it to 00000.jpg, ..."""
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    url = presign(f"{S3_BASE}_{ANGLE}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0),
         "-i", url, "-t", str(dur),
         "-vf", f"field=top,scale=1920:1080,fps={fps}",
         "-an", "-q:v", "2", "-start_number", "0",
         str(frame_dir / "%05d.jpg")],
        check=True,
    )
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit("ffmpeg produced no frames")
    return frames


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def mask_foot(mask: np.ndarray) -> Optional[XY]:
    """Bottom-centre pixel of a boolean mask (foot point), or None."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    y_max = int(ys.max())
    band = xs[ys >= y_max - 3]            # bottom 4 rows -> robust centre
    return float(band.mean()), float(y_max)


def mask_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
@dataclass
class TrackState:
    """Per-obj court trail for the fading streak on the map."""
    trail: List[XY] = field(default_factory=list)


def draw_left(frame: np.ndarray,
              masks: Dict[int, np.ndarray]) -> np.ndarray:
    """FR frame with each obj's mask outline + id, persistent colour."""
    out = frame.copy()
    overlay = frame.copy()
    for oid, m in masks.items():
        col = palette(oid)
        overlay[m] = col
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, col, 2, cv2.LINE_AA)
        bb = mask_bbox(m)
        if bb is not None:
            x1, y1 = int(bb[0]), int(bb[1])
            cv2.putText(out, f"#{oid}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
    return out


def draw_court(base: np.ndarray, pts: Dict[int, XY],
               trails: Dict[int, TrackState]) -> np.ndarray:
    canvas = base.copy()
    for oid, st in trails.items():
        col = palette(oid)
        tl = st.trail
        for k in range(1, len(tl)):
            a = k / len(tl)
            p0 = court.cm_to_canvas(tl[k - 1][0], tl[k - 1][1], SCALE, PAD)
            p1 = court.cm_to_canvas(tl[k][0], tl[k][1], SCALE, PAD)
            cv2.line(canvas, p0, p1,
                     (int(col[0] * a), int(col[1] * a), int(col[2] * a)),
                     2, cv2.LINE_AA)
    for oid, (cx, cy) in pts.items():
        col = palette(oid)
        px, py = court.cm_to_canvas(cx, cy, SCALE, PAD)
        cv2.circle(canvas, (px, py), 11, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(oid), (px - 6, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
#  Detection (RF-DETR)
# --------------------------------------------------------------------------- #
def detect_players(model, frame_bgr: np.ndarray, thr: float) -> np.ndarray:
    """RF-DETR player boxes (xyxy float32) on one BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    dets = model.predict(rgb, threshold=thr)
    if len(dets) == 0:
        return np.zeros((0, 4), np.float32)
    keep = np.isin(dets.class_id, list(PLAYER_CLASSES))
    return dets.xyxy[keep].astype(np.float32)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t0", type=float, default=1400.0)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="demo/sam2_track_demo.mp4")
    ap.add_argument("--conf", type=float, default=0.45,
                    help="RF-DETR player confidence (higher -> drop the "
                         "tiny/spurious distant boxes SAM2 can't hold)")
    ap.add_argument("--seed-search", type=int, default=12,
                    help="scan the first N frames and seed SAM2 on the one "
                         "with the most confident player detections (a more "
                         "complete object set than frame 0 alone)")
    ap.add_argument("--max-objects", type=int, default=12,
                    help="cap seeded objects (SAM2 per-object cost on MPS is "
                         "steep; 12 covers a full 5v5 + refs)")
    ap.add_argument("--min-box-h", type=float, default=55.0,
                    help="drop seed boxes shorter than this (distant/partial "
                         "players SAM2 cannot hold a stable mask on)")
    ap.add_argument("--min-mask-px", type=int, default=400,
                    help="ignore obj masks smaller than this many pixels "
                         "(lost / collapsed objects)")
    ap.add_argument("--court-margin", type=float, default=120.0,
                    help="cm tolerance just outside court bounds")
    ap.add_argument("--x-inset", type=float, default=40.0,
                    help="require feet >= this many cm inside both baselines")
    ap.add_argument("--trail-len", type=int, default=10)
    ap.add_argument("--device", default="auto",
                    choices=("auto", "mps", "cpu"))
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the extracted /tmp frame dir (default: purge)")
    a = ap.parse_args(argv)

    import torch
    from rfdetr import RFDETRSmall
    from sam2.build_sam import build_sam2_video_predictor

    if a.device == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = a.device
    L, W = court.COURT_LENGTH_CM, court.COURT_WIDTH_CM
    Hinv = np.linalg.inv(np.load(CAL / f"{ANGLE}_c2a_H.npy"))

    print(f"[1/5] extract FR frames  t0={a.t0} dur={a.dur} fps={a.fps}")
    frames = extract_frames(a.t0, a.dur, a.fps, FRAME_DIR)
    n_frames = len(frames)
    print(f"      {n_frames} frames -> {FRAME_DIR}")

    print(f"[2/5] load RF-DETR + SAM2 (base+) on {dev}")
    rf = RFDETRSmall(pretrain_weights="demo/rfdetr_bball.pth")
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=dev)
    state = predictor.init_state(str(FRAME_DIR))

    # ---- pick the best seed frame, seed a bounded object set ------------ #
    print("[3/5] seed players (best frame in first "
          f"{a.seed_search} frames)")
    bgr_cache: Dict[int, np.ndarray] = {}

    def bgr(i: int) -> np.ndarray:
        if i not in bgr_cache:
            bgr_cache[i] = cv2.imread(str(frames[i]))
        return bgr_cache[i]

    def good_boxes(boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return boxes
        keep = (boxes[:, 3] - boxes[:, 1]) >= a.min_box_h
        return boxes[keep]

    # Seed on whichever of the first `seed_search` frames yields the most
    # full-height player boxes -- a single propagate pass cannot add
    # objects later (see module docstring), so the seed frame should be as
    # complete as possible.
    seed_idx, seed_boxes = 0, good_boxes(detect_players(rf, bgr(0), a.conf))
    for i in range(1, min(a.seed_search, n_frames)):
        b = good_boxes(detect_players(rf, bgr(i), a.conf))
        if len(b) > len(seed_boxes):
            seed_idx, seed_boxes = i, b

    # Largest boxes first (nearest / most reliable players), capped.
    order = np.argsort(-(seed_boxes[:, 3] - seed_boxes[:, 1]))
    seed_boxes = seed_boxes[order][: a.max_objects]
    next_id = 0
    for b in seed_boxes:
        predictor.add_new_points_or_box(state, frame_idx=seed_idx,
                                        obj_id=next_id, box=b)
        next_id += 1
    print(f"      seed frame {seed_idx}: {next_id} players")
    if next_id == 0:
        raise SystemExit("no player detections to seed -- lower --conf")

    # ---- propagate (forward from seed; reverse fills the pre-seed gap) -- #
    print(f"[4/5] propagate {next_id} objects through {n_frames} frames")
    base_court = court.draw_topdown_court(scale=SCALE, pad=PAD)
    ch, cw = base_court.shape[:2]
    fh, fw = bgr(0).shape[:2]
    left_w = int(OUT_H * fw / fh)
    panel_w = int(OUT_H * cw / ch)
    title_h = 40
    out_w, out_h = left_w + panel_w, OUT_H + title_h

    trails: Dict[int, TrackState] = {}
    t_start = time.time()
    seen_ids: set = set()
    # frames may arrive seed..n-1 (forward) then seed-1..0 (reverse), so we
    # buffer JPEG-encoded panels keyed by index and mux in order at the end
    # (JPEG keeps the buffer ~100MB for 720 frames vs ~2GB raw).
    panels: Dict[int, bytes] = {}

    def render(fidx: int, obj_ids, mask_logits) -> int:
        ml = mask_logits.cpu().numpy()               # (N,1,H,W) logits
        masks_px: Dict[int, np.ndarray] = {}
        pts_cm: Dict[int, XY] = {}
        for k, oid in enumerate(obj_ids):
            m = ml[k, 0] > 0.0
            if int(m.sum()) < a.min_mask_px:
                continue
            oid = int(oid)
            masks_px[oid] = m
            foot = mask_foot(m)
            if foot is None:
                continue
            fp = np.array([[[foot[0], foot[1]]]], np.float64)
            cx, cy = cv2.perspectiveTransform(fp, Hinv)[0, 0]
            if not (a.x_inset <= cx <= L - a.x_inset):
                continue
            if not (-a.court_margin <= cy <= W + a.court_margin):
                continue
            pts_cm[oid] = (float(cx), float(cy))
            seen_ids.add(oid)
            st = trails.setdefault(oid, TrackState())
            st.trail.append((float(cx), float(cy)))
            st.trail[:] = st.trail[-a.trail_len:]

        left = cv2.resize(draw_left(bgr(fidx), masks_px), (left_w, OUT_H))
        right = cv2.resize(draw_court(base_court, pts_cm, trails),
                           (panel_w, OUT_H), interpolation=cv2.INTER_AREA)
        title = np.zeros((title_h, out_w, 3), np.uint8)
        cv2.putText(title,
                    f"UBALL  -  SAM2 mask-propagation tracking (FR)"
                    f"   |   {len(pts_cm)} players on court",
                    (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panel = np.vstack([title, np.hstack([left, right])])
        _, enc = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        panels[fidx] = enc.tobytes()
        bgr_cache.pop(fidx, None)
        return len(pts_cm)

    autocast = (torch.autocast(device_type="mps", dtype=torch.bfloat16)
                if dev == "mps" else torch.inference_mode())
    with torch.inference_mode(), autocast:
        for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            n_on = render(fidx, obj_ids, mask_logits)
            if len(panels) % 60 == 0:
                el = time.time() - t_start
                print(f"      {len(panels)}/{n_frames} frames  "
                      f"({len(panels) / el:.2f} fps, {n_on} on court)")
        # reverse-fill any frames before the seed frame
        if seed_idx > 0:
            for tr in trails.values():       # trails are forward-only; reset
                tr.trail.clear()
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(
                    state, start_frame_idx=seed_idx, reverse=True):
                if fidx not in panels:
                    render(fidx, obj_ids, mask_logits)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(a.out, fourcc, a.fps, (out_w, out_h))
    for fidx in range(n_frames):
        if fidx in panels:
            vw.write(cv2.imdecode(np.frombuffer(panels[fidx], np.uint8),
                                  cv2.IMREAD_COLOR))
    vw.release()

    el = time.time() - t_start
    print(f"[5/5] done: {len(panels)} frames in {el / 60:.1f} min "
          f"-> {len(panels) / el:.2f} fps")
    print(f"      tracked obj ids seen on court: {len(seen_ids)} "
          f"({sorted(seen_ids)})")
    print(f"      output: {Path(a.out).resolve()}")

    if not a.keep_frames and FRAME_DIR.exists():
        shutil.rmtree(FRAME_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
