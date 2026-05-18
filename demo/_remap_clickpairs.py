#!/usr/bin/env python3
"""Remap every camera's clickpair court-cm coords from the old NBA model
to the new court_2.dxf model via a piecewise-linear warp through the
court features shared by both models (baseline, FT line, center,
opposite FT, opposite baseline on X; baseline, lane edge, center, lane
edge, sideline on Y). Each named feature maps exactly; points placed
between features (relative to nearby painted lines) interpolate
smoothly. Image pixels are untouched. Originals -> *_clickpairs.nba.json
"""
import json
from pathlib import Path

import numpy as np

CAL = Path("demo/calibration")
CAMS = ["FL_cal", "FR_cal", "NL_cal", "NR_cal"]

# old NBA model -> new CAD model control points
XO = [0.0, 579.0, 1432.5, 2286.0, 2865.0]
XN = [0.0, 594.3, 1071.85, 1549.4, 2143.7]
YO = [0.0, 518.0, 762.0, 1006.0, 1524.0]
YN = [0.0, 518.9, 713.2, 907.5, 1426.4]


def remap(xc: float, yc: float):
    return float(np.interp(xc, XO, XN)), float(np.interp(yc, YO, YN))


for cam in CAMS:
    p = CAL / f"{cam}_clickpairs.json"
    if not p.exists():
        print(f"skip {cam} (no clickpairs)")
        continue
    bak = CAL / f"{cam}_clickpairs.nba.json"
    if not bak.exists():
        bak.write_text(p.read_text())
    pairs = json.loads(bak.read_text())["pairs"]   # always remap from NBA orig
    out = [[a, list(remap(b[0], b[1]))] for a, b in pairs]
    p.write_text(json.dumps({"pairs": out}, indent=2))
    print(f"{cam}: {len(out)} pairs remapped  "
          f"(e.g. {pairs[0][1]} -> {[round(v,1) for v in out[0][1]]})")
