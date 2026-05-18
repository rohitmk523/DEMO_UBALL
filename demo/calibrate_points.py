#!/usr/bin/env python3
"""
Numbered-point calibration (replaces cryptic landmark names).

Workflow the operator asked for:

  1. We define a DENSE set of court points at *known* cm coords, labelled
     1..N (numbers, not names like "L_lane_base_top").
  2. `--ref` renders a single side-by-side window:
       LEFT  = the real camera frame, with the CURRENT (imperfect) homography
               drawn as ORANGE hollow guesses + numbers.
       RIGHT = the top-down court diagram, with GREEN solid dots + the same
               numbers.
     The operator only has to say "number 18 is actually at (x, y)" for the
     points that are wrong (e.g. FL center / center-circle).
  3. Put the confirmed picks in a JSON ({"points": {"18": [px, py], ...}})
     and re-run with `--confirm picks.json` to redraw (confirmed = RED solid),
     then add `--solve` to recompute + cache the homography.

Only operator-confirmed points feed the homography solve — the orange
guesses are visual aid only (using them would be circular).

  python demo/calibrate_points.py --name NL_cal --ref
  python demo/calibrate_points.py --name FL_cal --ref
  python demo/calibrate_points.py --name FL_cal --confirm picks_fl.json --solve

Outputs (demo/calibration/):
  <name>_points_ref.jpg      side-by-side reference (left frame / right court)
  <name>_points_legend.txt   number -> court description + cm
  on --solve:
  <name>_H.npy  <name>_calibration.json  <name>_points.json
  <name>_overlay.jpg  <name>_topdown.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.calibration_integration import CalibrationIntegration  # noqa: E402
from lib import court  # noqa: E402
from calibrate_homography import _court_polylines, _topdown  # noqa: E402

CALIB_DIR = Path("demo/calibration")
FRAME_DIR = Path("demo/frames")

L = court.COURT_LENGTH_CM        # 2865 (X, baseline->baseline)
W = court.COURT_WIDTH_CM         # 1524 (Y, sideline->sideline)
CX, CY = court.CENTER_X, court.CENTER_Y          # 1432.5, 762
LH = court.LANE_HALF             # 244  (lane half-width)
FT = court.FT_DISTANCE_CM        # 579  (baseline -> FT line)
HOOP = court.HOOP_FROM_BASELINE_CM  # 160
TOK = court.THREE_R_CM           # 723.9 (3pt radius -> top-of-key apex)

# number -> ((x_cm, y_cm), human description). All are crisp painted-line
# intersections or well-defined circle apexes, spread across the whole court.
# X=0 is the AMG / NL-hoop baseline; X=2865 is the far / FL-bench baseline.
POINTS: Dict[int, Tuple[Tuple[float, float], str]] = {
    # --- Left baseline (X=0) : the line FL sees but NL cannot ---
    1:  ((0.0, 0.0),            "L baseline @ scoreboard corner"),
    2:  ((0.0, 381.0),          "L baseline, scoreboard quarter"),
    3:  ((0.0, CY - LH),        "L baseline @ lane edge (top)"),
    4:  ((0.0, CY),             "L baseline midpoint (under NL hoop)"),
    5:  ((0.0, CY + LH),        "L baseline @ lane edge (bot)"),
    6:  ((0.0, 1143.0),         "L baseline, Iverson quarter"),
    7:  ((0.0, W),              "L baseline @ Iverson corner"),
    # --- Left key / free-throw (X=579) ---
    8:  ((FT, CY - LH),         "L FT line @ lane (top)"),
    9:  ((FT, CY),              "L FT-line center"),
    10: ((FT, CY + LH),         "L FT line @ lane (bot)"),
    11: ((FT + LH, CY),         "L FT-circle far apex"),
    12: ((HOOP + TOK, CY),      "L top-of-key (3pt apex)"),
    # --- Sideline quarter marks ---
    13: ((716.0, 0.0),          "scoreboard sideline, L quarter"),
    14: ((716.0, W),            "Iverson sideline, L quarter"),
    15: ((2148.0, 0.0),         "scoreboard sideline, R quarter"),
    16: ((2148.0, W),           "Iverson sideline, R quarter"),
    # --- Center ---
    17: ((CX, 0.0),             "center line @ scoreboard sideline"),
    18: ((CX, CY - court.CENTER_CIRCLE_R_CM), "center circle TOP"),
    19: ((CX, CY),              "center court"),
    20: ((CX, CY + court.CENTER_CIRCLE_R_CM), "center circle BOT"),
    21: ((CX, W),               "center line @ Iverson sideline"),
    22: ((CX - court.CENTER_CIRCLE_R_CM, CY), "center circle LEFT apex"),
    23: ((CX + court.CENTER_CIRCLE_R_CM, CY), "center circle RIGHT apex"),
    # --- Right key / free-throw (X=2286) ---
    24: ((L - HOOP - TOK, CY),  "R top-of-key (3pt apex)"),
    25: ((L - FT - LH, CY),     "R FT-circle far apex"),
    26: ((L - FT, CY - LH),     "R FT line @ lane (top)"),
    27: ((L - FT, CY),          "R FT-line center"),
    28: ((L - FT, CY + LH),     "R FT line @ lane (bot)"),
    # --- Right baseline (X=2865) : far / FL-bench end ---
    29: ((L, 0.0),              "R baseline @ scoreboard corner"),
    30: ((L, 381.0),            "R baseline, scoreboard quarter"),
    31: ((L, CY - LH),          "R baseline @ lane edge (top)"),
    32: ((L, CY),               "R baseline midpoint (under far hoop)"),
    33: ((L, CY + LH),          "R baseline @ lane edge (bot)"),
    34: ((L, 1143.0),           "R baseline, Iverson quarter"),
    35: ((L, W),                "R baseline @ Iverson corner"),
}

GREEN = (0, 200, 0)
RED = (0, 0, 235)
ORANGE = (0, 165, 255)


def _label(img: np.ndarray, n: int, x: int, y: int, color, *,
           filled: bool) -> None:
    """Draw a numbered marker; filled=confirmed, hollow=guess."""
    if filled:
        cv2.circle(img, (x, y), 6, color, -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.circle(img, (x, y), 7, color, 2, cv2.LINE_AA)
    txt = str(n)
    org = (x + 8, y - 8)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)


def _court_panel(scale: float = 0.42, pad: int = 50) -> np.ndarray:
    """Top-down court with every numbered point in GREEN."""
    canvas = court.draw_topdown_court(scale=scale, pad=pad)
    for n, ((xc, yc), _desc) in POINTS.items():
        px, py = court.cm_to_canvas(xc, yc, scale, pad)
        _label(canvas, n, px, py, GREEN, filled=True)
    cv2.putText(canvas, "COURT (green = point number)", (pad, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
    return canvas


def _frame_panel(frame: np.ndarray,
                 guess: CalibrationIntegration | None,
                 confirmed: Dict[int, Tuple[int, int]]) -> np.ndarray:
    """Frame with RED confirmed picks and ORANGE current-H guesses."""
    out = frame.copy()
    for n, ((xc, yc), _desc) in POINTS.items():
        if n in confirmed:
            x, y = confirmed[n]
            _label(out, n, int(x), int(y), RED, filled=True)
        elif guess is not None:
            x, y = guess.court_to_image(xc, yc)
            if -2000 < x < frame.shape[1] + 2000 and \
               -2000 < y < frame.shape[0] + 2000:
                _label(out, n, int(x), int(y), ORANGE, filled=False)
    cv2.putText(out, "REAL FRAME  red=confirmed  orange=current-H guess",
                (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                2, cv2.LINE_AA)
    return out


def _side_by_side(frame_panel: np.ndarray,
                  court_panel: np.ndarray, h: int = 950) -> np.ndarray:
    def fit(img: np.ndarray) -> np.ndarray:
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h),
                          interpolation=cv2.INTER_AREA)

    a, b = fit(frame_panel), fit(court_panel)
    div = np.full((h, 6, 3), 60, dtype=np.uint8)
    return np.hstack([a, div, b])


def _legend_text(name: str) -> str:
    rows = [f"{name} — numbered calibration points "
            f"(X=length 0..{int(L)}cm, Y=width 0..{int(W)}cm)\n"]
    for n, ((xc, yc), desc) in POINTS.items():
        rows.append(f"{n:>3} : ({xc:7.1f},{yc:7.1f}) cm  {desc}")
    return "\n".join(rows) + "\n"


def _load_guess(name: str) -> CalibrationIntegration | None:
    cf = CALIB_DIR / f"{name}_calibration.json"
    if not cf.exists():
        return None
    c = CalibrationIntegration()
    return c if c.load_calibration(str(cf)) else None


def _read_confirmed(path: Path) -> Dict[int, Tuple[int, int]]:
    data = json.loads(path.read_text())
    pts = data.get("points", data)
    out: Dict[int, Tuple[int, int]] = {}
    for k, v in pts.items():
        n = int(k)
        if n not in POINTS:
            raise SystemExit(f"point {n} not in 1..{max(POINTS)}")
        out[n] = (int(v[0]), int(v[1]))
    return out


def _overlay(frame: np.ndarray, calib: CalibrationIntegration,
             confirmed: Dict[int, Tuple[int, int]]) -> np.ndarray:
    out = frame.copy()
    for poly in _court_polylines():
        pts = calib.court_to_image_batch(poly)
        for p, q in zip(pts, pts[1:]):
            cv2.line(out, p, q, (0, 255, 0), 2, cv2.LINE_AA)
    for n, ((xc, yc), _d) in POINTS.items():
        px, py = calib.court_to_image(xc, yc)
        cv2.circle(out, (px, py), 4, (0, 255, 0), -1)
        if n in confirmed:
            cx, cy = confirmed[n]
            cv2.circle(out, (cx, cy), 5, RED, -1)
            cv2.line(out, (px, py), (cx, cy), RED, 1, cv2.LINE_AA)
    return out


def _solve(name: str, frame_path: Path, frame: np.ndarray,
           confirmed: Dict[int, Tuple[int, int]]) -> None:
    if len(confirmed) < 4:
        raise SystemExit(f"need >=4 confirmed points to solve, "
                         f"got {len(confirmed)}")
    court_pts = [POINTS[n][0] for n in sorted(confirmed)]
    image_pts = [(float(confirmed[n][0]), float(confirmed[n][1]))
                 for n in sorted(confirmed)]
    calib = CalibrationIntegration()
    if not calib.compute_homography(court_pts, image_pts):
        raise SystemExit("homography failed (degenerate / collinear points?)")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    calib.save_calibration(str(CALIB_DIR / f"{name}_calibration.json"))
    np.save(CALIB_DIR / f"{name}_H.npy", calib.homography_matrix)
    (CALIB_DIR / f"{name}_points.json").write_text(json.dumps(
        {"points": {str(n): list(confirmed[n]) for n in sorted(confirmed)}},
        indent=2))
    cv2.imwrite(str(CALIB_DIR / f"{name}_overlay.jpg"),
                _overlay(frame, calib, confirmed))
    cv2.imwrite(str(CALIB_DIR / f"{name}_topdown.jpg"),
                _topdown(frame, calib))

    errs = []
    for (xc, yc), (ix, iy) in zip(court_pts, image_pts):
        px, py = calib.court_to_image(xc, yc)
        errs.append(((px - ix) ** 2 + (py - iy) ** 2) ** 0.5)
    print(f"{name}: solved from {len(confirmed)} pts | reproj "
          f"mean={np.mean(errs):.1f}px max={np.max(errs):.1f}px")
    print(f"  -> {CALIB_DIR}/{name}_H.npy + _calibration.json "
          f"+ _overlay.jpg + _topdown.jpg")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True,
                   help="stem, e.g. NL_cal (frame demo/frames/<name>.jpg)")
    p.add_argument("--frame", type=Path,
                   help="override frame path (default demo/frames/<name>.jpg)")
    p.add_argument("--confirm", type=Path,
                   help='picks JSON {"points":{"7":[px,py],...}}')
    p.add_argument("--ref", action="store_true",
                   help="render side-by-side reference (default if no confirm)")
    p.add_argument("--solve", action="store_true",
                   help="recompute + cache homography from confirmed points")
    a = p.parse_args(argv)

    frame_path = a.frame or (FRAME_DIR / f"{a.name}.jpg")
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise SystemExit(f"cannot read frame {frame_path}")

    confirmed = _read_confirmed(a.confirm) if a.confirm else {}
    guess = _load_guess(a.name)

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    legend_p = CALIB_DIR / f"{a.name}_points_legend.txt"
    legend_p.write_text(_legend_text(a.name))

    ref = _side_by_side(_frame_panel(frame, guess, confirmed),
                        _court_panel())
    ref_p = CALIB_DIR / f"{a.name}_points_ref.jpg"
    cv2.imwrite(str(ref_p), ref)
    print(f"{a.name}: ref -> {ref_p}  ({ref.shape[1]}x{ref.shape[0]})")
    print(f"  legend -> {legend_p}  | confirmed={len(confirmed)} "
          f"| guess-H={'yes' if guess else 'NONE'}")

    if a.solve:
        _solve(a.name, frame_path, frame, confirmed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
