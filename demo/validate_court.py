#!/usr/bin/env python3
"""
Single-camera court-mapping validation (NL-primary decision, 2026-05-18).

Calibration showed NL alone covers the whole court (~19px) while FL is poor
(~64px). So the demo court map is single-camera NL — no fusion. This is the
real-footage check:

  1. pull a deinterlaced NL frame at --t (S3, no full download)
  2. YOLOv11 person detection
  3. project each foot point (bbox bottom-center) through the NL homography
     (vendored CalibrationIntegration) -> court CM
  4. render [ NL frame + boxes/foot pts | top-down court + dots ]

  python demo/validate_court.py --t 1400 \
      --calib demo/calibration/NL_cal_calibration.json

Output: demo/validation/<game>_<angle>_t<t>_courtcheck.jpg  (+ stats)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_frame import extract  # noqa: E402
from lib import court  # noqa: E402
from lib.calibration_integration import CalibrationIntegration  # noqa: E402
from lib.dual_camera_fusion import foot_point  # noqa: E402

VAL_DIR = Path("demo/validation")
WEIGHT_CANDIDATES = [
    Path("../models/yolo11m.pt"),
    Path("../uball_court_mapping/yolo11m.pt"),
    Path("../Uball_tracking/yolo11m.pt"),
]


def _find_weights(explicit: str | None) -> str:
    if explicit:
        return explicit
    for c in WEIGHT_CANDIDATES:
        if c.exists():
            return str(c)
    raise SystemExit("no YOLO weights found; pass --weights")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="c2a354fe")
    p.add_argument("--angle", default="NL")
    p.add_argument("--t", type=float, required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--conf", type=float, default=0.35)
    a = p.parse_args(argv)

    from ultralytics import YOLO

    calib = CalibrationIntegration(a.calib)
    if calib.inverse_homography is None:
        raise SystemExit(f"{a.calib} has no homography")
    model = YOLO(_find_weights(a.weights))

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "f.jpg"
        extract(a.angle, a.t, f)               # deinterlaced by default
        img = cv2.imread(str(f))
        res = model.predict(str(f), classes=[0], conf=a.conf, verbose=False)

    boxes = [tuple(float(v) for v in b.xyxy[0].tolist())
             for b in res[0].boxes]
    print(f"{a.angle} t={a.t}: {len(boxes)} people detected")

    # frame panel
    fr = img.copy()
    court_pts = []
    for bb in boxes:
        x1, y1, x2, y2 = (int(v) for v in bb)
        cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        fx, fy = foot_point(bb)
        cv2.circle(fr, (int(fx), int(fy)), 6, (0, 0, 255), -1)
        court_pts.append(calib.image_to_court(fx, fy))

    # court panel
    scale, pad = 0.34, 40
    cp = court.draw_topdown_court(scale=scale, pad=pad)
    on, off = 0, 0
    for (cx, cy) in court_pts:
        if -200 <= cx <= court.COURT_LENGTH_CM + 200 and \
           -200 <= cy <= court.COURT_WIDTH_CM + 200:
            on += 1
        else:
            off += 1
        px, py = court.cm_to_canvas(cx, cy, scale, pad)
        px = int(np.clip(px, 0, cp.shape[1] - 1))
        py = int(np.clip(py, 0, cp.shape[0] - 1))
        cv2.circle(cp, (px, py), 9, (0, 220, 0), -1)
        cv2.circle(cp, (px, py), 9, (255, 255, 255), 1)
    print(f"  on/near court: {on}   off court (bad): {off}")

    H = 600
    def fit(im):
        return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H))
    def lbl(im, t):
        cv2.rectangle(im, (0, 0), (im.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(im, t, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        return im
    sheet = np.hstack([
        lbl(fit(fr), f"{a.angle}  {len(boxes)} ppl  (red=foot pt)"),
        lbl(fit(cp), f"COURT  on={on} off={off}"),
    ])
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    out = VAL_DIR / f"{a.game}_{a.angle}_t{int(a.t)}_courtcheck.jpg"
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
