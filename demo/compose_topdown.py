#!/usr/bin/env python3
"""
Overall correctness check: warp ALL calibrated cameras onto ONE H court.

For each camera with a fitted TPS (<name>_tps.npz from solve_fisheye), warp
its real frame onto the shared top-down court diagram, masked to the region
its click-points actually support (convex hull + margin). Blend the four,
draw the model court lines on top.

Reading it:
  * Painted lines from each camera should land ON the white model lines.
  * In overlap zones (e.g. center), lines from two cameras should COINCIDE.
    Doubled / ghosted lines there = those two calibrations disagree.
  * Each camera tints its own region so you can see who covers what and
    that FL+FR+NL+NR together tile the whole court.

  python demo/compose_topdown.py                 # all 4 default cams
  python demo/compose_topdown.py --cams FL_cal NL_cal

Output: demo/calibration/_composite_topdown.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib import court  # noqa: E402

CALIB_DIR = Path("demo/calibration")
FRAME_DIR = Path("demo/frames")
DEFAULT = ["FL_cal", "FR_cal", "NL_cal", "NR_cal"]
TINT = {"FL_cal": (80, 0, 0), "FR_cal": (0, 60, 0),
        "NL_cal": (0, 0, 80), "NR_cal": (0, 50, 50)}
SCALE, PAD = 0.30, 40


def compose(cams: List[str]) -> None:
    diagram = court.draw_topdown_court(scale=SCALE, pad=PAD)
    dh, dw = diagram.shape[:2]
    gx, gy = np.meshgrid(np.arange(dw), np.arange(dh))
    cmx = ((gx - PAD) / SCALE).ravel()
    cmy = ((gy - PAD) / SCALE).ravel()
    court_pts = np.stack([cmx, cmy], 1).astype(np.float64)
    on_court = ((cmx >= 0) & (cmx <= court.COURT_LENGTH_CM) &
                (cmy >= 0) & (cmy <= court.COURT_WIDTH_CM)).reshape(dh, dw)

    acc = np.zeros((dh, dw, 3), np.float32)
    cnt = np.zeros((dh, dw), np.float32)
    used = []
    for name in cams:
        hp = CALIB_DIR / f"{name}_H.npy"
        fr = FRAME_DIR / f"{name}.jpg"
        if not hp.exists() or not fr.exists():
            print(f"skip {name} (no H/frame yet)")
            continue
        H = np.load(hp)
        frame = cv2.imread(str(fr))
        fhh, fww = frame.shape[:2]
        src = cv2.perspectiveTransform(
            court_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        mapx = src[:, 0].reshape(dh, dw).astype(np.float32)
        mapy = src[:, 1].reshape(dh, dw).astype(np.float32)
        # this camera supports a canvas px iff it maps inside the court AND
        # the source lands inside that camera's actual image
        sup = (on_court & (mapx >= 0) & (mapx < fww) &
               (mapy >= 0) & (mapy < fhh))
        warp = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        warp[~sup] = 0
        tint = np.array(TINT.get(name, (40, 40, 40)), np.float32)
        acc += warp.astype(np.float32) + sup[..., None] * tint
        cnt += sup
        used.append(name)

    cnt = np.maximum(cnt, 1)
    blended = (acc / cnt[..., None]).clip(0, 255).astype(np.uint8)
    out = cv2.addWeighted(blended, 0.85, diagram, 0.6, 0)
    p = CALIB_DIR / "_composite_topdown.jpg"
    cv2.imwrite(str(p), out)
    print(f"composite from {used} -> {p}  ({dw}x{dh})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", nargs="+", default=DEFAULT)
    a = ap.parse_args(argv)
    compose(a.cams)
    return 0


if __name__ == "__main__":
    sys.exit(main())
