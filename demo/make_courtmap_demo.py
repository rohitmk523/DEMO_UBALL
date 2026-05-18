#!/usr/bin/env python3
"""
Investor demo: side-by-side  [ real game video | top-down court map ].

YOLO detects players in the camera frame; each player's foot point is
mapped through our calibrated homography (<name>_H.npy, court_cm ->
image_px) into court space and drawn as a red dot on the CAD court
diagram (lib/court.py). As players move in the left video the red dots
move with them on the right.

Frames are deinterlaced exactly as the calibration frames were
(field=top,scale=1920:1080) so pixel space matches the homography.

  python demo/make_courtmap_demo.py \
      --video /path/2026-05-05_..._FR.mp4 --name FR_cal \
      --t0 2400 --dur 300 --fps 15 --out demo/courtmap_demo.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib import court  # noqa: E402

CAL = Path("demo/calibration")
SCALE, PAD = 0.42, 50          # bigger top-down for a crisp demo
OUT_H = 720                     # side-by-side panel height


def extract_segment(video: str, t0: float, dur: float, fps: int) -> Path:
    """ffmpeg -> deinterlaced segment (matches calibration pixel space)."""
    tmp = Path(tempfile.mkdtemp()) / "seg.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0),
         "-i", video, "-t", str(dur),
         "-vf", f"field=top,scale=1920:1080,fps={fps}",
         "-an", "-q:v", "3", str(tmp)],
        check=True)
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", default="FR_cal")
    ap.add_argument("--t0", type=float, default=2400)
    ap.add_argument("--dur", type=float, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="demo/courtmap_demo.mp4")
    ap.add_argument("--weights", default="yolo11l.pt",
                    help="YOLO11 large: more reliable than n/m for the demo")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--roi-top", type=float, default=0.22,
                    help="ignore detections whose feet are above this "
                         "fraction of frame height (bench/far wall + the "
                         "camera's least-accurate far zone)")
    a = ap.parse_args()

    H = np.load(CAL / f"{a.name}_H.npy")
    Hinv = np.linalg.inv(H)
    L, W = court.COURT_LENGTH_CM, court.COURT_WIDTH_CM
    base = court.draw_topdown_court(scale=SCALE, pad=PAD)
    ch, cw = base.shape[:2]

    from ultralytics import YOLO
    model = YOLO(a.weights)

    print(f"extracting {a.dur}s @ {a.fps}fps from t={a.t0} ...")
    seg = extract_segment(a.video, a.t0, a.dur, a.fps)
    cap = cv2.VideoCapture(str(seg))

    vid_w = int(OUT_H * 1920 / 1080)
    panel_cw = int(OUT_H * cw / ch)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(a.out, fourcc, a.fps, (vid_w + panel_cw, OUT_H))

    n = total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        res = model.predict(frame, classes=[0], conf=a.conf,
                            verbose=False, device="mps")[0]
        canvas = base.copy()
        cnt = 0
        M = 40  # cm tolerance just outside the lines (baseline play)
        roi_y = a.roi_top * frame.shape[0]
        for box in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            if (y2 - y1) < 55:           # tiny box = distant spectator/bench
                continue
            if y2 < roi_y:               # bench / far wall / inaccurate zone
                continue
            foot = np.array([[[(x1 + x2) / 2.0, y2]]], np.float64)
            cx, cy = cv2.perspectiveTransform(foot, Hinv)[0, 0]
            if -M <= cx <= L + M and -M <= cy <= W + M:   # strictly on court
                px, py = court.cm_to_canvas(cx, cy, SCALE, PAD)
                cv2.circle(canvas, (px, py), 10, (0, 0, 255), -1)
                cv2.circle(canvas, (px, py), 10, (255, 255, 255), 1,
                           cv2.LINE_AA)
                cnt += 1
        total += cnt

        left = cv2.resize(frame, (vid_w, OUT_H))
        right = cv2.resize(canvas, (panel_cw, OUT_H),
                           interpolation=cv2.INTER_AREA)
        for img, txt in ((left, "LIVE CAMERA (FR)"),
                         (right, f"COURT MAP  -  {cnt} players")):
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(img, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(np.hstack([left, right]))
        if n % 100 == 0:
            print(f"  {n} frames  ({cnt} players this frame)")

    cap.release()
    vw.release()
    print(f"done: {n} frames, avg "
          f"{total / max(n,1):.1f} dots/frame -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
