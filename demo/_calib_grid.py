#!/usr/bin/env python3
"""Annotate a frame with a labelled pixel grid so landmark image
coordinates can be read off precisely (headless calibration helper).

  python demo/_calib_grid.py demo/frames/FR_cal.jpg demo/frames/_grid_FR.jpg
"""
import sys

import cv2
import numpy as np


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    im = cv2.imread(src)
    if im is None:
        raise SystemExit(f"cannot read {src}")
    h, w = im.shape[:2]
    ov = im.copy()
    # fine 50px grid (dim), coarse 100px grid (brighter) + labels
    for x in range(0, w, 50):
        cv2.line(ov, (x, 0), (x, h), (60, 60, 60), 1)
    for y in range(0, h, 50):
        cv2.line(ov, (0, y), (w, y), (60, 60, 60), 1)
    for x in range(0, w, 100):
        cv2.line(ov, (x, 0), (x, h), (0, 180, 255), 1)
    for y in range(0, h, 100):
        cv2.line(ov, (0, y), (w, y), (0, 180, 255), 1)
    for x in range(0, w, 100):
        for y in range(0, h, 100):
            cv2.putText(ov, f"{x},{y}", (x + 2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(ov, f"{x},{y}", (x + 2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1,
                        cv2.LINE_AA)
    blend = cv2.addWeighted(im, 0.45, ov, 0.55, 0)
    cv2.imwrite(dst, blend)
    print(f"{dst}  ({w}x{h})")


if __name__ == "__main__":
    main()
