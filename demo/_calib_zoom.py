#!/usr/bin/env python3
"""Zoom a rectangular region of a frame with an absolute-pixel ruler so a
landmark's full-image coordinates can be read precisely.

  python demo/_calib_zoom.py <frame> <x0> <y0> <x1> <y1> <out> [zoom]
"""
import sys

import cv2
import numpy as np


def main() -> None:
    a = sys.argv
    frame, x0, y0, x1, y1, out = a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5]), a[6]
    z = float(a[7]) if len(a) > 7 else 4.0
    im = cv2.imread(frame)
    if im is None:
        raise SystemExit(f"cannot read {frame}")
    H, W = im.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = im[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    bh, bw = big.shape[:2]
    # ruler every 25 src px (minor) / 50 src px (major, labelled)
    for sx in range(x0 - x0 % 25, x1, 25):
        px = int((sx - x0) * z)
        major = sx % 50 == 0
        cv2.line(big, (px, 0), (px, bh), (0, 180, 255) if major else (50, 50, 50), 1)
        if major:
            cv2.putText(big, str(sx), (px + 2, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(big, str(sx), (px + 2, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
    for sy in range(y0 - y0 % 25, y1, 25):
        py = int((sy - y0) * z)
        major = sy % 50 == 0
        cv2.line(big, (0, py), (bw, py), (0, 180, 255) if major else (50, 50, 50), 1)
        if major:
            cv2.putText(big, str(sy), (2, py + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(big, str(sy), (2, py + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out, big)
    print(f"{out}  src[{x0}:{x1},{y0}:{y1}] x{z}")


if __name__ == "__main__":
    main()
