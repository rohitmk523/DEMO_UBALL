#!/usr/bin/env python3
"""
RANSAC homography calibration from click-pairs (headless).

Mirrors the PROVEN method from uball_court_mapping
(app/services/calibration.py:42) that already works for FR/FL:

    H, mask = cv2.findHomography(court_pts, video_pts, cv2.RANSAC, 5.0)

Feed ALL clicked pairs; RANSAC keeps the geometrically consistent
inliers and rejects the rest (their shipped calibration: 7 inliers of
15). No TPS, no custom distortion term, no manual point pruning — those
were detours away from the approach that works. A plane homography is a
pinhole model: it cannot correct anamorphic SuperView footage (a
capture-time lens-standardisation problem, not a math one).

  python demo/solve_fisheye.py --name FL_cal      # FL / FR / NL / NR

Outputs: <name>_overlay.jpg  <name>_topdown.jpg
         <name>_calibration.json  (homography_matrix, inliers/outliers)
         <name>_H.npy             (court_cm -> image_px, 3x3)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib import court  # noqa: E402
from calibrate_homography import _court_polylines  # noqa: E402

CALIB_DIR = Path("demo/calibration")
FRAME_DIR = Path("demo/frames")


def _densify(poly, step_cm: float = 12.0) -> np.ndarray:
    P = np.asarray(poly, np.float64)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step_cm))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.array(out)


def _load_pairs(name: str):
    p = CALIB_DIR / f"{name}_clickpairs.json"
    if not p.exists():
        raise SystemExit(f"no {p} — run calibrate_click.py --name {name}")
    pairs = json.loads(p.read_text())["pairs"]
    if len(pairs) < 4:
        raise SystemExit(f"need >=4 pairs, have {len(pairs)}")
    img = np.array([a for a, _ in pairs], np.float32)
    cm = np.array([b for _, b in pairs], np.float32)
    return img, cm


def solve(name: str, frame_path: Path) -> None:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise SystemExit(f"cannot read {frame_path}")
    h, w = frame.shape[:2]
    img, cm = _load_pairs(name)

    # PROVEN method (uball_court_mapping/app/services/calibration.py:42)
    H, mask = cv2.findHomography(cm, img, cv2.RANSAC, 5.0)
    if H is None:
        raise SystemExit("homography failed (need 4+ non-collinear pairs)")
    mask = mask.ravel().astype(bool)
    Hinv = np.linalg.inv(H)

    pr = cv2.perspectiveTransform(
        cm.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    errs = np.linalg.norm(pr - img, axis=1)
    in_e = errs[mask]

    # ---- overlay: project ALL court lines, each segment clipped to image
    ov = frame.copy()
    for poly in _court_polylines():
        ip = cv2.perspectiveTransform(
            _densify(poly).reshape(-1, 1, 2), H).reshape(-1, 2).astype(int)
        for a, b in zip(ip[:-1], ip[1:]):
            ok, p1, p2 = cv2.clipLine((0, 0, w, h), tuple(a), tuple(b))
            if ok:
                cv2.line(ov, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
    for (ix, iy), inl in zip(img.astype(int), mask):
        cv2.circle(ov, (int(ix), int(iy)), 5,
                   (0, 200, 0) if inl else (0, 0, 255), -1)
    cv2.imwrite(str(CALIB_DIR / f"{name}_overlay.jpg"), ov)

    # ---- top-down: warp frame onto the court diagram ----
    scale, pad = 0.30, 40
    diagram = court.draw_topdown_court(scale=scale, pad=pad)
    dh, dw = diagram.shape[:2]
    S = np.array([[scale, 0, pad], [0, scale, pad], [0, 0, 1]])
    warped = cv2.warpPerspective(frame, S @ Hinv, (dw, dh))
    cv2.imwrite(str(CALIB_DIR / f"{name}_topdown.jpg"),
                cv2.addWeighted(warped, 0.6, diagram, 0.7, 0))

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CALIB_DIR / f"{name}_H.npy", H)
    (CALIB_DIR / f"{name}_calibration.json").write_text(json.dumps({
        "homography_matrix": H.tolist(),
        "image_size": [w, h],
        "num_points": int(len(errs)),
        "inliers": int(mask.sum()),
        "outliers": int((~mask).sum()),
    }, indent=2))
    print(f"{name}: {len(errs)} pairs | RANSAC inliers={mask.sum()} "
          f"outliers={(~mask).sum()} | inlier reproj "
          f"mean={in_e.mean():.1f}px max={in_e.max():.1f}px")
    print(f"  -> {CALIB_DIR}/{name}_overlay.jpg + _topdown.jpg + _H.npy")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True,
                   help="FL_cal / FR_cal / NL_cal / NR_cal")
    p.add_argument("--frame", type=Path)
    a = p.parse_args(argv)
    fp = a.frame or (FRAME_DIR / f"{a.name}.jpg")
    solve(a.name, fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
