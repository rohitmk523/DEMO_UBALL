#!/usr/bin/env python3
"""
Real-footage validation of the dual-camera fusion (06 §10).

NO CLICKING NEEDED — runs the instant the operator has produced the two
per-camera calibration JSONs (Step 1 / calibrate_dual). It:

  1. pulls a synchronized FL+NL frame-pair at timestamp --t (S3, no full DL)
  2. runs YOLOv11 person detection on each (reuse pattern; local weights)
  3. feeds per-camera detections to lib.dual_camera_fusion.DualCameraFusion
  4. renders a 3-panel check image:
       [ FL frame + boxes/foot pts | NL frame + boxes | top-down court + dots ]
     dot colour: green = seen by BOTH cameras (overlap merged),
                 blue  = FL only,  orange = NL only

Use it to eyeball whether real players land at sensible court positions and
whether a player in the overlap shows as ONE green dot (calibrations agree),
not two. That is the real-footage fusion validation.

  python demo/validate_dual.py --game c2a354fe --t 1400 \
      --fl-calib demo/calibration/FL_cal_calibration.json \
      --nl-calib demo/calibration/NL_cal_calibration.json

Output: demo/validation/<game>_t<t>_dualcheck.jpg  (+ printed stats)
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
from lib.dual_camera_fusion import (  # noqa: E402
    DualCameraFusion, RawDetection, foot_point, FL_CAM_ID, NL_CAM_ID,
)

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
    raise SystemExit("no YOLO weights found; pass --weights path/to/yolo11m.pt")


def _detect_people(model, frame_path: Path, conf: float) -> list[RawDetection]:
    res = model.predict(str(frame_path), classes=[0], conf=conf, verbose=False)
    dets: list[RawDetection] = []
    for i, b in enumerate(res[0].boxes):
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        dets.append(RawDetection(track_id=i,
                                  bbox=(x1, y1, x2, y2),
                                  confidence=float(b.conf[0])))
    return dets


def _draw_boxes(img: np.ndarray, dets: list[RawDetection],
                color: tuple) -> np.ndarray:
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        fx, fy = foot_point(d.bbox)
        cv2.circle(out, (int(fx), int(fy)), 5, color, -1)  # projected point
    return out


def _panel_label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return img


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="c2a354fe")
    p.add_argument("--t", type=float, required=True, help="timestamp seconds")
    p.add_argument("--fl-calib", required=True)
    p.add_argument("--nl-calib", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--max-dist-cm", type=float, default=150.0)
    a = p.parse_args(argv)

    from ultralytics import YOLO  # heavy import, do it after arg parse

    fusion = DualCameraFusion(a.fl_calib, a.nl_calib,
                              max_player_distance_cm=a.max_dist_cm)
    model = YOLO(_find_weights(a.weights))

    with tempfile.TemporaryDirectory() as td:
        fl_f = Path(td) / "fl.jpg"
        nl_f = Path(td) / "nl.jpg"
        extract("FL", a.t, fl_f)
        extract("NL", a.t, nl_f)
        fl_img, nl_img = cv2.imread(str(fl_f)), cv2.imread(str(nl_f))
        fl_dets = _detect_people(model, fl_f, a.conf)
        nl_dets = _detect_people(model, nl_f, a.conf)

    fused = fusion.fuse(fl_dets, nl_dets)
    n_both = sum(1 for f in fused if len(f.cameras) == 2)
    n_fl = sum(1 for f in fused if f.cameras == [FL_CAM_ID])
    n_nl = sum(1 for f in fused if f.cameras == [NL_CAM_ID])
    print(f"FL det={len(fl_dets)}  NL det={len(nl_dets)}  -> fused={len(fused)} "
          f"(both={n_both}, FL-only={n_fl}, NL-only={n_nl})")

    # court panel
    scale, pad = 0.34, 40
    cpanel = court.draw_topdown_court(scale=scale, pad=pad)
    for f in fused:
        cx, cy = court.cm_to_canvas(f.court_xy[0], f.court_xy[1], scale, pad)
        col = ((0, 220, 0) if len(f.cameras) == 2
               else (255, 130, 0) if f.cameras == [FL_CAM_ID]
               else (0, 165, 255))
        cv2.circle(cpanel, (cx, cy), 9, col, -1)
        cv2.circle(cpanel, (cx, cy), 9, (255, 255, 255), 1)

    H = 540
    def _fit(im):
        return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H))
    fl_v = _panel_label(_fit(_draw_boxes(fl_img, fl_dets, (255, 130, 0))),
                        f"FL  ({len(fl_dets)} ppl)")
    nl_v = _panel_label(_fit(_draw_boxes(nl_img, nl_dets, (0, 165, 255))),
                        f"NL  ({len(nl_dets)} ppl)")
    ct_v = _panel_label(_fit(cpanel),
                        f"COURT  green=both({n_both}) blue=FL orange=NL")
    sheet = np.hstack([fl_v, nl_v, ct_v])

    VAL_DIR.mkdir(parents=True, exist_ok=True)
    out = VAL_DIR / f"{a.game}_t{int(a.t)}_dualcheck.jpg"
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
