#!/usr/bin/env python3
"""
03_DEMO_BUILD_PLAN.md — Step 1: establish + cache the court homography.

Two modes:

  Interactive (run locally, needs a GUI / cv2.imshow window):
    python demo/calibrate_homography.py --frame demo/frames/c2a354fe_FL_t1400.jpg \
        --interactive
  Click each visible court landmark when prompted; coords are saved to a
  correspondence JSON you can reuse / hand-tune.

  Config (headless, reproducible — CI-friendly):
    python demo/calibrate_homography.py --frame ... --config demo/calibration/c2a354fe_FL_corr.json

Correspondence JSON format:
  { "landmarks": { "<name from court.LANDMARKS_CM>": [px_x, px_y], ... } }

Outputs (into demo/calibration/):
  <stem>_calibration.json   homography + points (CalibrationIntegration format)
  <stem>_H.npy              the 3x3 court_cm -> image_px matrix (cached)
  <stem>_overlay.jpg        court lines reprojected onto the frame (quality check)
  <stem>_topdown.jpg        frame warped to a top-down court (quality check)

The homography math is the vendored CalibrationIntegration (verbatim reuse).
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

CALIB_DIR = Path("demo/calibration")


def _court_polylines() -> List[List[Tuple[float, float]]]:
    """Court lines as cm polylines, for reprojection quality check."""
    L, W = court.COURT_LENGTH_CM, court.COURT_WIDTH_CM
    cx, cy = court.CENTER_X, court.CENTER_Y
    lh = court.LANE_HALF
    lines: List[List[Tuple[float, float]]] = [
        [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],          # boundary
        [(cx, 0), (cx, W)],                                  # center line
    ]
    # center circle + ft circles as sampled arcs
    for (ccx, ccy, r) in [(cx, cy, court.CENTER_CIRCLE_R_CM),
                          (court.FT_DISTANCE_CM, cy, lh),
                          (L - court.FT_DISTANCE_CM, cy, lh)]:
        arc = [(ccx + r * np.cos(t), ccy + r * np.sin(t))
               for t in np.linspace(0, 2 * np.pi, 48)]
        lines.append(arc)
    # keys
    for bx, s in ((0.0, 1.0), (L, -1.0)):
        fx = bx + s * court.FT_DISTANCE_CM
        lines.append([(bx, cy - lh), (fx, cy - lh),
                      (fx, cy + lh), (bx, cy + lh)])
    return lines


def _draw_overlay(frame: np.ndarray, calib: CalibrationIntegration) -> np.ndarray:
    """Reproject court lines + named landmarks onto the frame."""
    out = frame.copy()
    for poly in _court_polylines():
        pts = calib.court_to_image_batch(poly)
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 255, 0), 2, cv2.LINE_AA)
    for name, (xc, yc) in court.LANDMARKS_CM.items():
        px, py = calib.court_to_image(xc, yc)
        cv2.circle(out, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(out, name, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def _topdown(frame: np.ndarray, calib: CalibrationIntegration,
             scale: float = 0.30, pad: int = 40) -> np.ndarray:
    """Warp frame to a top-down court and blend with the court diagram."""
    diagram = court.draw_topdown_court(scale=scale, pad=pad)
    h, w = diagram.shape[:2]
    # image_px -> canvas_px : scale&pad @ inverse(court->image)
    S = np.array([[scale, 0, pad], [0, scale, pad], [0, 0, 1]], dtype=np.float64)
    M = S @ calib.inverse_homography
    warped = cv2.warpPerspective(frame, M, (w, h))
    return cv2.addWeighted(warped, 0.6, diagram, 0.7, 0)


def _interactive(frame: np.ndarray) -> Dict[str, List[int]]:
    names = list(court.LANDMARKS_CM.keys())
    picks: Dict[str, List[int]] = {}
    idx = [0]
    disp = frame.copy()

    def redraw() -> None:
        d = frame.copy()
        for nm, (x, y) in picks.items():
            cv2.circle(d, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(d, nm, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1)
        if idx[0] < len(names):
            cv2.putText(d, f"CLICK: {names[idx[0]]}  (s=skip, u=undo, q=done)",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
        disp[:] = d

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and idx[0] < len(names):
            picks[names[idx[0]]] = [x, y]
            idx[0] += 1
            redraw()

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", on_mouse)
    redraw()
    while True:
        cv2.imshow("calibrate", disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27) or idx[0] >= len(names):
            break
        if k == ord("s") and idx[0] < len(names):
            idx[0] += 1
            redraw()
        if k == ord("u") and idx[0] > 0:
            idx[0] -= 1
            picks.pop(names[idx[0]], None)
            redraw()
    cv2.destroyAllWindows()
    return picks


def calibrate(frame_path: Path, landmarks_px: Dict[str, List[int]]) -> None:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise SystemExit(f"cannot read frame {frame_path}")

    court_pts: List[Tuple[float, float]] = []
    image_pts: List[Tuple[float, float]] = []
    for name, px in landmarks_px.items():
        if name not in court.LANDMARKS_CM:
            raise SystemExit(f"unknown landmark {name!r}; valid: "
                             f"{list(court.LANDMARKS_CM)}")
        court_pts.append(court.LANDMARKS_CM[name])
        image_pts.append((float(px[0]), float(px[1])))

    if len(court_pts) < 4:
        raise SystemExit(f"need >=4 correspondences, got {len(court_pts)}")

    calib = CalibrationIntegration()
    if not calib.compute_homography(court_pts, image_pts):
        raise SystemExit("homography computation failed (degenerate points?)")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    stem = frame_path.stem
    calib.save_calibration(str(CALIB_DIR / f"{stem}_calibration.json"))
    np.save(CALIB_DIR / f"{stem}_H.npy", calib.homography_matrix)
    (CALIB_DIR / f"{stem}_corr.json").write_text(
        json.dumps({"landmarks": landmarks_px}, indent=2))

    cv2.imwrite(str(CALIB_DIR / f"{stem}_overlay.jpg"),
                _draw_overlay(frame, calib))
    cv2.imwrite(str(CALIB_DIR / f"{stem}_topdown.jpg"),
                _topdown(frame, calib))

    # reprojection error on the input correspondences
    errs = []
    for (xc, yc), (ix, iy) in zip(court_pts, image_pts):
        px, py = calib.court_to_image(xc, yc)
        errs.append(((px - ix) ** 2 + (py - iy) ** 2) ** 0.5)
    print(f"{stem}: {len(court_pts)} pts, "
          f"reproj err mean={np.mean(errs):.1f}px max={np.max(errs):.1f}px")
    print(f"  -> {CALIB_DIR}/{stem}_H.npy + _calibration.json + "
          f"_overlay.jpg + _topdown.jpg")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--interactive", action="store_true")
    g.add_argument("--config", type=Path,
                   help="correspondence JSON {landmarks:{name:[px,py]}}")
    a = p.parse_args(argv)

    if a.interactive:
        frame = cv2.imread(str(a.frame))
        if frame is None:
            raise SystemExit(f"cannot read frame {a.frame}")
        landmarks = _interactive(frame)
    else:
        data = json.loads(a.config.read_text())
        landmarks = data["landmarks"]

    calibrate(a.frame, landmarks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
