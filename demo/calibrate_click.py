#!/usr/bin/env python3
"""
Dead-simple click-pair calibration. No landmark names, no numbers.

  TOP    = the real court (camera frame)
  BOTTOM = the H court (top-down diagram you are mapping onto)

Click a spot on the REAL court (top), then click the SAME spot on the
H court (bottom). That's one correspondence pair. Add as many as you
like, anywhere you like. Press `s` -> homography solved + overlay saved.

A magnifier loupe follows the cursor on whichever panel you're over, so
clicks stay pixel-accurate even though the panels are scaled to fit.

Keys:
  s  solve + save (H.npy, calibration.json, corr.json, overlay, topdown)
  u  undo last pair      r  reset all      o  (re)open overlay
  q / ESC  save pairs + quit

  python demo/calibrate_click.py --name FL_cal
  python demo/calibrate_click.py --name NL_cal   # etc. (FR_cal / NR_cal)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.calibration_integration import CalibrationIntegration  # noqa: E402
from lib import court  # noqa: E402
from calibrate_homography import _court_polylines, _topdown  # noqa: E402

CALIB_DIR = Path("demo/calibration")
FRAME_DIR = Path("demo/frames")
MAXH, GAP = 940, 8
CS, PAD = 0.30, 40                       # court diagram scale / pad (cm->px)
PALETTE = [(0, 0, 255), (0, 255, 0), (255, 128, 0), (0, 255, 255),
           (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 165, 255)]


class ClickCalib:
    def __init__(self, name: str, frame_path: Path) -> None:
        self.name = name
        self.frame_path = frame_path
        self.frame = cv2.imread(str(frame_path))
        if self.frame is None:
            raise SystemExit(f"cannot read frame {frame_path}")
        self.fh, self.fw = self.frame.shape[:2]
        self.court_cv = court.draw_topdown_court(scale=CS, pad=PAD)
        self.ch, self.cw = self.court_cv.shape[:2]

        # fit BOTH stacked panels on screen (same display width)
        r = (MAXH - GAP) / (self.fh / self.fw + self.ch / self.cw)
        self.dw = int(min(r, 1280))
        self.fdh = int(round(self.dw * self.fh / self.fw))
        self.cdh = int(round(self.dw * self.ch / self.cw))
        self.rf = self.dw / self.fw                    # frame_px -> disp
        self.rc = self.dw / self.cw                    # court_px -> disp

        self.L = court.COURT_LENGTH_CM
        self.W = court.COURT_WIDTH_CM
        self.cz = 0.10                                  # center-zone inset
        self.pairs: List[Tuple[Tuple[float, float],
                               Tuple[float, float]]] = []  # (img_px, cm)
        self.pending: Optional[Tuple[float, float]] = None  # img_px
        self.cursor: Tuple[int, int] = (0, 0)
        self._load()

        self.win = f"click-calib [{name}]"
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self._on_mouse)

    # ---- coord maps ------------------------------------------------------
    def _court_canvas_to_cm(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx - PAD) / CS, (cy - PAD) / CS

    def _zone_cm(self) -> Tuple[float, float, float, float]:
        return (self.L * self.cz, self.W * self.cz,
                self.L * (1 - self.cz), self.W * (1 - self.cz))

    def _in_zone(self, xc: float, yc: float) -> bool:
        x0, y0, x1, y1 = self._zone_cm()
        return x0 <= xc <= x1 and y0 <= yc <= y1

    def _cm_to_court_disp(self, x: float, y: float) -> Tuple[int, int]:
        return (int(round((x * CS + PAD) * self.rc)),
                int(round((y * CS + PAD) * self.rc)) + self.fdh + GAP)

    # ---- persistence -----------------------------------------------------
    def _corr_path(self) -> Path:
        return CALIB_DIR / f"{self.name}_clickpairs.json"

    def _load(self) -> None:
        p = self._corr_path()
        if p.exists():
            d = json.loads(p.read_text())
            self.pairs = [((float(a[0]), float(a[1])),
                           (float(b[0]), float(b[1])))
                          for a, b in d.get("pairs", [])]

    def _save_pairs(self) -> None:
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        self._corr_path().write_text(json.dumps(
            {"pairs": [[list(a), list(b)] for a, b in self.pairs]}, indent=2))

    # ---- mouse -----------------------------------------------------------
    def _on_mouse(self, ev: int, x: int, y: int, _f: int, _p) -> None:
        self.cursor = (x, y)
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        if y < self.fdh:                                   # REAL court panel
            self.pending = (x / self.rf, y / self.rf)
        elif y >= self.fdh + GAP:                           # H court panel
            if self.pending is None:
                print("click the REAL court (top) first, then here")
                return
            cm = self._court_canvas_to_cm((x) / self.rc,
                                          (y - self.fdh - GAP) / self.rc)
            self.pairs.append((self.pending, cm))
            self.pending = None

    # ---- render ----------------------------------------------------------
    def _loupe(self, canvas: np.ndarray, src: np.ndarray,
               sx: float, sy: float, at: Tuple[int, int]) -> None:
        z, half = 5, 36
        h, w = src.shape[:2]
        x0, y0 = int(np.clip(sx - half, 0, w - 1)), int(np.clip(sy - half, 0, h - 1))
        x1, y1 = min(w, x0 + 2 * half), min(h, y0 + 2 * half)
        crop = src[y0:y1, x0:x1]
        if crop.size == 0:
            return
        loup = cv2.resize(crop, (2 * half * z, 2 * half * z),
                          interpolation=cv2.INTER_NEAREST)
        cx, cy = int((sx - x0) * z), int((sy - y0) * z)
        cv2.drawMarker(loup, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 30, 1)
        cv2.rectangle(loup, (0, 0), (loup.shape[1] - 1, loup.shape[0] - 1),
                      (0, 255, 255), 2)
        ax, ay = at
        lh, lw = loup.shape[:2]
        ax = min(ax, canvas.shape[1] - lw)
        ay = min(ay, canvas.shape[0] - lh)
        canvas[ay:ay + lh, ax:ax + lw] = loup

    def _canvas(self) -> np.ndarray:
        fp = cv2.resize(self.frame, (self.dw, self.fdh),
                        interpolation=cv2.INTER_AREA)
        cp = cv2.resize(self.court_cv, (self.dw, self.cdh),
                        interpolation=cv2.INTER_AREA)
        zx0, zy0, zx1, zy1 = self._zone_cm()              # center-zone box
        cv2.rectangle(cp,
                      (int((zx0 * CS + PAD) * self.rc),
                       int((zy0 * CS + PAD) * self.rc)),
                      (int((zx1 * CS + PAD) * self.rc),
                       int((zy1 * CS + PAD) * self.rc)),
                      (255, 200, 0), 2, cv2.LINE_AA)
        n_in = sum(self._in_zone(xc, yc) for _, (xc, yc) in self.pairs)
        for i, ((ix, iy), (xc, yc)) in enumerate(self.pairs):
            col = PALETTE[i % len(PALETTE)]
            inz = self._in_zone(xc, yc)
            fx, fy = int(ix * self.rf), int(iy * self.rf)
            cv2.circle(fp, (fx, fy), 6, col, -1 if inz else 2)
            cv2.circle(fp, (fx, fy), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(fp, str(i + 1), (fx + 7, fy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            ccx = int((xc * CS + PAD) * self.rc)
            ccy = int((yc * CS + PAD) * self.rc)
            cv2.circle(cp, (ccx, ccy), 6, col, -1 if inz else 2)
            cv2.circle(cp, (ccx, ccy), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(cp, str(i + 1), (ccx + 7, ccy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        if self.pending is not None:
            px, py = int(self.pending[0] * self.rf), int(self.pending[1] * self.rf)
            cv2.drawMarker(fp, (px, py), (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 22, 2)
            cv2.putText(fp, "now click same spot on H court below",
                        (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
        msg = (f"{self.name}  pairs={len(self.pairs)} (in-zone={n_in})  "
               f"{'PICK on H court (bottom)' if self.pending else 'click REAL court (top)'}"
               f"   s=solve  u=undo  r=reset  -/+=zone  o=overlay  q=quit")
        cv2.putText(fp, msg, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(fp, msg, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cp, f"H COURT  yellow box = CENTER ZONE "
                    f"(margin {self.cz:.0%}); hollow dots dropped from solve",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 200, 0), 2, cv2.LINE_AA)

        gap = np.full((GAP, self.dw, 3), 50, dtype=np.uint8)
        canvas = np.vstack([fp, gap, cp])
        cxp, cyp = self.cursor
        if 0 <= cyp < self.fdh:
            self._loupe(canvas, self.frame, cxp / self.rf, cyp / self.rf,
                        (self.dw - 360, 8))
        elif cyp >= self.fdh + GAP:
            self._loupe(canvas, self.court_cv, cxp / self.rc,
                        (cyp - self.fdh - GAP) / self.rc, (self.dw - 360,
                        self.fdh + GAP + 8))
        return canvas

    # ---- solve -----------------------------------------------------------
    def _solve(self) -> None:
        self._save_pairs()
        used = [p for p in self.pairs if self._in_zone(*p[1])]
        if len(used) < 4:
            print(f"need >=4 IN-ZONE pairs, have {len(used)} "
                  f"(of {len(self.pairs)}); widen zone with '-' or add pairs")
            return
        if len(used) < len(self.pairs):
            print(f"center-zone: using {len(used)}/{len(self.pairs)} pairs "
                  f"(dropped {len(self.pairs) - len(used)} edge pairs)")
        court_pts = [(xc, yc) for _, (xc, yc) in used]
        image_pts = [(ix, iy) for (ix, iy), _ in used]
        calib = CalibrationIntegration()
        if not calib.compute_homography(court_pts, image_pts):
            print("homography failed (need 4+ non-collinear pairs)")
            return
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        calib.save_calibration(str(CALIB_DIR / f"{self.name}_calibration.json"))
        np.save(CALIB_DIR / f"{self.name}_H.npy", calib.homography_matrix)
        (CALIB_DIR / f"{self.name}_corr.json").write_text(json.dumps(
            {"pairs": [[list(a), list(b)] for a, b in self.pairs]}, indent=2))

        ov = self.frame.copy()
        for poly in _court_polylines():
            pts = calib.court_to_image_batch(poly)
            for a, b in zip(pts, pts[1:]):
                cv2.line(ov, a, b, (0, 255, 0), 2, cv2.LINE_AA)
        for (ix, iy), (xc, yc) in self.pairs:
            solid = self._in_zone(xc, yc)
            cv2.circle(ov, (int(ix), int(iy)), 5, (0, 0, 255),
                       -1 if solid else 2)
        cv2.imwrite(str(CALIB_DIR / f"{self.name}_overlay.jpg"), ov)
        cv2.imwrite(str(CALIB_DIR / f"{self.name}_topdown.jpg"),
                    _topdown(self.frame, calib))
        errs = [((calib.court_to_image(xc, yc)[0] - ix) ** 2 +
                 (calib.court_to_image(xc, yc)[1] - iy) ** 2) ** 0.5
                for (xc, yc), (ix, iy) in zip(court_pts, image_pts)]
        print(f"{self.name}: {len(used)}/{len(self.pairs)} in-zone pairs  "
              f"reproj mean={np.mean(errs):.1f}px max={np.max(errs):.1f}px "
              f"-> {CALIB_DIR}/{self.name}_overlay.jpg")

    def _open_overlay(self) -> None:
        ov = CALIB_DIR / f"{self.name}_overlay.jpg"
        if not ov.exists():
            print("no overlay yet — press s")
            return
        img = cv2.imread(str(ov))
        s = self.fdh / img.shape[0]
        cv2.imshow(f"overlay [{self.name}]",
                   cv2.resize(img, (int(img.shape[1] * s), self.fdh)))

    # ---- loop ------------------------------------------------------------
    def run(self) -> None:
        print(f"[{self.name}] click real(top)+H(bottom); s=solve q=quit")
        while True:
            cv2.imshow(self.win, self._canvas())
            k = cv2.waitKey(16) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                self._solve()
                self._open_overlay()
            elif k == ord("u") and self.pairs:
                self.pairs.pop()
                self.pending = None
            elif k == ord("r"):
                self.pairs.clear()
                self.pending = None
            elif k in (ord("-"), ord("_")):
                self.cz = round(max(0.0, self.cz - 0.02), 2)
            elif k in (ord("="), ord("+")):
                self.cz = round(min(0.30, self.cz + 0.02), 2)
            elif k == ord("o"):
                self._open_overlay()
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                break
        self._save_pairs()
        cv2.destroyAllWindows()
        print(f"saved {self._corr_path()}  ({len(self.pairs)} pairs)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True,
                   help="output stem, e.g. FL_cal / NL_cal / FR_cal / NR_cal")
    p.add_argument("--frame", type=Path,
                   help="override (default demo/frames/<name>.jpg)")
    a = p.parse_args(argv)
    fp = a.frame or (FRAME_DIR / f"{a.name}.jpg")
    ClickCalib(a.name, fp).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
