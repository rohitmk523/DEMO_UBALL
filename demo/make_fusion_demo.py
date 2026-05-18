#!/usr/bin/env python3
"""
Investor demo: 4-camera fusion player tracking.

LEFT  : 2x2 grid of the 4 live camera feeds (FL, FR, NL, NR).
RIGHT : one top-down court map (lib/court.draw_topdown_court) with ONE
        smoothed red dot per real player + a short fading trail.

Pipeline per synchronized 4-cam frame:
  1. Each camera runs its OWN ultralytics tracker (separate track-id state).
  2. Foot point = bbox bottom-center -> court cm via that cam's inv(H).
  3. Region-aware trust: drop detections off-court or outside the
     court-X band that camera "owns".
  4. Cross-camera merge: greedy mutual NN cluster within `merge_cm` so a
     player seen by 2+ cameras becomes ONE fused point (mean position).
  5. Temporal smoothing: persistent fused tracks, EMA position, debounced
     birth, delayed death. Only smoothed tracks are drawn (kills flicker).

Frames are deinterlaced (field=top,scale=1920:1080) so pixel space
matches the homographies, exactly as the calibration frames were.

  python demo/make_fusion_demo.py --t0 1400 --dur 240 --fps 24
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib import court  # noqa: E402

CAL = Path("demo/calibration")
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
                 tmpdir: Path) -> Path:
    """Pull a deinterlaced segment for one angle (HTTP range, no full DL)."""
    url = presign(f"{S3_BASE}_{angle}.mp4")
    out = tmpdir / f"{angle}.mp4"
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
#  Cross-camera per-frame merge
# --------------------------------------------------------------------------- #
def merge_points(pts: List[XY], merge_cm: float) -> List[XY]:
    """Greedy mutual nearest-neighbour cluster; cluster -> mean position."""
    n = len(pts)
    if n <= 1:
        return list(pts)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
            if d < merge_cm:
                pairs.append((d, i, j))
    pairs.sort()
    for _d, i, j in pairs:
        parent[find(i)] = find(j)

    clusters: Dict[int, List[XY]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(pts[i])
    return [(float(np.mean([p[0] for p in c])),
             float(np.mean([p[1] for p in c]))) for c in clusters.values()]


# --------------------------------------------------------------------------- #
#  Temporal smoothing
# --------------------------------------------------------------------------- #
@dataclass
class SmoothTrack:
    pos: XY
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    trail: List[XY] = field(default_factory=list)


class Smoother:
    """Persistent fused tracks: EMA position, debounced birth, lazy death."""

    def __init__(self, match_cm: float, ema_new: float,
                 birth_hits: int, death_misses: int, trail_len: int):
        self.match_cm = match_cm
        self.ema_new = ema_new
        self.birth_hits = birth_hits
        self.death_misses = death_misses
        self.trail_len = trail_len
        self.tracks: List[SmoothTrack] = []

    def update(self, points: List[XY]) -> List[SmoothTrack]:
        cands: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            for pi, p in enumerate(points):
                d = float(np.hypot(tr.pos[0] - p[0], tr.pos[1] - p[1]))
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
            nx, ny = points[pi]
            a = self.ema_new
            tr.pos = (tr.pos[0] * (1 - a) + nx * a,
                      tr.pos[1] * (1 - a) + ny * a)
            tr.hits += 1
            tr.misses = 0
            if tr.hits >= self.birth_hits:
                tr.confirmed = True
            tr.trail.append(tr.pos)
            tr.trail[:] = tr.trail[-self.trail_len:]

        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
        for pi, p in enumerate(points):
            if pi not in used_p:
                self.tracks.append(SmoothTrack(pos=p, trail=[p]))

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
    offset: int             # per-cam integer-second sync nudge


def court_points(boxes: np.ndarray, cam: CamCfg, min_box_h: float,
                 L: float, W: float, margin: float,
                 roi_y: float, x_inset: float) -> List[XY]:
    pts: List[XY] = []
    for x1, y1, x2, y2 in boxes:
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
        pts.append((float(cx), float(cy)))
    return pts


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


def draw_court(base: np.ndarray, tracks: List[SmoothTrack]) -> np.ndarray:
    canvas = base.copy()
    for tr in tracks:
        tl = tr.trail
        for k in range(1, len(tl)):
            a = k / len(tl)
            p0 = court.cm_to_canvas(tl[k - 1][0], tl[k - 1][1], SCALE, PAD)
            p1 = court.cm_to_canvas(tl[k][0], tl[k][1], SCALE, PAD)
            cv2.line(canvas, p0, p1,
                     (0, int(120 * a), int(255 * a)), 2, cv2.LINE_AA)
    for tr in tracks:
        px, py = court.cm_to_canvas(tr.pos[0], tr.pos[1], SCALE, PAD)
        cv2.circle(canvas, (px, py), 10, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


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
    ap.add_argument("--conf", type=float, default=0.40)
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
    # trusted court-X band fractions (of court length L).
    # Derived empirically from a per-camera projection diagnostic on the
    # t1400 synced frames: FL & NL reliably cover the LOW-X half (X~0..1050);
    # FR reliably covers the HIGH-X half (X~1600..2143); NR is the noisiest
    # and is restricted to the low/mid band where it agrees with FL/NL.
    ap.add_argument("--fl-band", default="0.0,0.55")
    ap.add_argument("--nl-band", default="0.0,0.55")
    ap.add_argument("--fr-band", default="0.52,1.0")
    ap.add_argument("--nr-band", default="0.0,0.62")
    a = ap.parse_args(argv)

    L, W = court.COURT_LENGTH_CM, court.COURT_WIDTH_CM
    offs = [int(x) for x in a.per_cam_offsets.split(",")]
    bands = {"FL": a.fl_band, "FR": a.fr_band,
             "NL": a.nl_band, "NR": a.nr_band}
    cams: Dict[str, CamCfg] = {}
    for i, ang in enumerate(ANGLES):
        lo_f, hi_f = (float(v) for v in bands[ang].split(","))
        cams[ang] = CamCfg(
            angle=ang,
            Hinv=np.linalg.inv(np.load(CAL / f"{ang}_c2a_H.npy")),
            band_lo=lo_f * L, band_hi=hi_f * L,
            offset=offs[i],
        )

    from ultralytics import YOLO
    models = {ang: YOLO(a.weights) for ang in ANGLES}

    tmpdir = Path(tempfile.mkdtemp(prefix="fusion4_"))
    caps: Dict[str, cv2.VideoCapture] = {}
    try:
        for ang in ANGLES:
            t0 = a.t0 + cams[ang].offset
            print(f"pulling {ang} segment t0={t0} dur={a.dur} ...")
            seg = pull_segment(ang, t0, a.dur, a.fps, tmpdir)
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
            pooled: List[XY] = []
            for ang in ANGLES:
                ok, frame = caps[ang].read()
                if not ok:
                    frames[ang] = None
                    continue
                ok_any = True
                frames[ang] = frame
                res = models[ang].track(
                    frame, persist=True, classes=[0], conf=a.conf,
                    tracker="bytetrack.yaml", device="mps", verbose=False,
                )[0]
                if res.boxes is None or res.boxes.xyxy is None:
                    continue
                boxes = res.boxes.xyxy.cpu().numpy()
                roi_y = a.roi_top * frame.shape[0]
                pooled.extend(court_points(boxes, cams[ang],
                                           a.min_box_h, L, W,
                                           a.court_margin, roi_y,
                                           a.x_inset))
            if not ok_any:
                break
            n += 1

            fused = merge_points(pooled, a.merge_cm)
            shown = smoother.update(fused)
            total += len(shown)

            grid = build_grid(frames)
            court_img = cv2.resize(draw_court(base, shown),
                                   (panel_w, OUT_H),
                                   interpolation=cv2.INTER_AREA)
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
    finally:
        for c in caps.values():
            c.release()
        for f in tmpdir.glob("*"):
            f.unlink(missing_ok=True)
        tmpdir.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
