#!/usr/bin/env python3
"""Parse the facility court CAD (court_2.dxf) into exact court geometry.

DXF group codes used: 0=entity, 8=layer, 10/20=X/Y, 40=radius,
50/51=arc start/end deg. Emits constants for lib/court.py with the
court-rectangle corner shifted to origin (0,0)."""
import sys
from pathlib import Path

DXF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/rohitkale/Cellstrat/GitHub_Repositories/uball_court_mapping/court_2.dxf")


def parse(path: Path):
    lines = path.read_text().splitlines()
    i, ents = 0, []
    while i < len(lines) - 1:
        code, val = lines[i].strip(), lines[i + 1].strip()
        if code == "0":
            if val in ("POLYLINE", "LINE", "CIRCLE", "ARC"):
                ents.append({"type": val, "layer": None, "verts": [],
                             "x": None, "y": None, "x2": None, "y2": None,
                             "r": None, "a0": None, "a1": None})
            elif val == "VERTEX":
                ents[-1]["verts"].append([None, None])
        else:
            e = ents[-1] if ents else None
            if e is None:
                i += 2
                continue
            if code == "8":
                e["layer"] = val
            elif code == "10":
                (e["verts"][-1].__setitem__(0, float(val))
                 if e["type"] == "POLYLINE" and e["verts"]
                 else e.__setitem__("x", float(val)))
            elif code == "20":
                (e["verts"][-1].__setitem__(1, float(val))
                 if e["type"] == "POLYLINE" and e["verts"]
                 else e.__setitem__("y", float(val)))
            elif code == "11":
                e["x2"] = float(val)
            elif code == "21":
                e["y2"] = float(val)
            elif code == "40":
                e["r"] = float(val)
            elif code == "50":
                e["a0"] = float(val)
            elif code == "51":
                e["a1"] = float(val)
        i += 2
    return ents


ents = parse(DXF)
polys = [e for e in ents if e["type"] == "POLYLINE"]
circles = [e for e in ents if e["type"] == "CIRCLE"]
arcs = [e for e in ents if e["type"] == "ARC"]
lines_ = [e for e in ents if e["type"] == "LINE"]

# court rectangle = the COURT-layer polyline whose bbox is largest
court = max((p for p in polys if p["layer"] == "COURT"),
            key=lambda p: (max(v[0] for v in p["verts"]) - min(v[0] for v in p["verts"])))
xs = [v[0] for v in court["verts"]]
ys = [v[1] for v in court["verts"]]
OX, OY = min(xs), min(ys)
L = max(xs) - OX
W = max(ys) - OY
print(f"# origin shift (CAD->court): ({OX}, {OY})")
print(f"COURT_LENGTH_CM = {L:.1f}")
print(f"COURT_WIDTH_CM  = {W:.1f}")
print(f"CENTER_X = {L/2:.2f}   CENTER_Y = {W/2:.2f}")

for c in circles:
    print(f"# CIRCLE  c=({c['x']-OX:.1f},{c['y']-OY:.1f})  r={c['r']:.1f}")
for a in arcs:
    print(f"# ARC     c=({a['x']-OX:.1f},{a['y']-OY:.1f})  r={a['r']:.1f}"
          f"  ang[{a['a0']}..{a['a1']}]")
for ln in lines_:
    print(f"# LINE    ({ln['x']-OX:.1f},{ln['y']-OY:.1f})->"
          f"({ln['x2']-OX:.1f},{ln['y2']-OY:.1f})")
for p in polys:
    if p["layer"] == "COURT":
        vv = [(round(v[0]-OX, 1), round(v[1]-OY, 1)) for v in p["verts"]]
        print(f"# POLY    {vv}")
