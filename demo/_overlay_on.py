#!/usr/bin/env python3
"""Project the saved court model through <name>_H.npy onto an arbitrary
frame (validate that a fixed-camera homography transfers to another
timestamp). Usage:

  python demo/_overlay_on.py FR_cal demo/frames/c2a354fe_FR_t1400.jpg out.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_homography import _court_polylines  # noqa: E402


def densify(poly, step=12.0):
    P = np.asarray(poly, np.float64)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.array(out)


def main():
    name, frame_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    H = np.load(f"demo/calibration/{name}_H.npy")
    im = cv2.imread(frame_path)
    h, w = im.shape[:2]
    for poly in _court_polylines():
        ip = cv2.perspectiveTransform(
            densify(poly).reshape(-1, 1, 2), H).reshape(-1, 2).astype(int)
        for a, b in zip(ip[:-1], ip[1:]):
            ok, p1, p2 = cv2.clipLine((0, 0, w, h), tuple(a), tuple(b))
            if ok:
                cv2.line(im, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(out, im)
    print(f"{out}  ({w}x{h})")


if __name__ == "__main__":
    main()
