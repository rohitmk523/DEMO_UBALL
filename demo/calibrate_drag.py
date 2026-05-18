#!/usr/bin/env python3
"""
Interactive drag calibration — grab the numbered red dots and drop them
on the real painted lines; the homography re-solves from where you put
them.

Single window:
  LEFT  = real frame, numbered RED dots you can click-hold-drag.
          A magnifier loupe pops near the grabbed dot for pixel-precise
          placement even though the view is downscaled.
  RIGHT = static top-down court, GREEN dots = the same numbers so you
          always know which physical point each red dot is.

Dots start at the CURRENT homography's projection (or a resumed session),
so usually you only nudge a few. Points a camera can't see (e.g. NL's
near baseline) — right-click the dot to EXCLUDE it (drawn grey, ignored
by the solve). Right-click again to re-include.

Keys:
  s   solve + save  (H.npy, calibration.json, points.json, overlay, topdown)
  r   reset all dots to the current-H projection
  f   fit / reset zoom    +/-  zoom    arrows/drag-bg  pan
  o   (re)open the saved overlay in a second window
  h   toggle help     q / ESC  save points.json and quit

  python demo/calibrate_drag.py --name FL_cal
  python demo/calibrate_drag.py --name NL_cal
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
from calibrate_points import (  # noqa: E402
    POINTS, GREEN, RED, ORANGE, CALIB_DIR, FRAME_DIR,
    _court_panel, _load_guess, _solve, _legend_text,
)
from calibrate_homography import _court_polylines  # noqa: E402

GREY = (130, 130, 130)
CYAN = (255, 255, 0)
MAXW = 1760          # keep whole window on a laptop screen
MAXH = 940
COURT_W = 470        # compact court reference strip (right side)


class DragCalib:
    def __init__(self, name: str, frame_path: Path) -> None:
        self.name = name
        self.frame_path = frame_path
        self.frame = cv2.imread(str(frame_path))
        if self.frame is None:
            raise SystemExit(f"cannot read frame {frame_path}")
        self.fh, self.fw = self.frame.shape[:2]
        # fit the WHOLE window on screen: frame panel + compact court strip
        self.scale = min(MAXH / self.fh, (MAXW - COURT_W - 6) / self.fw)
        self.disp_h = int(round(self.fh * self.scale))
        self.court = cv2.resize(_court_panel(), (COURT_W, self.disp_h),
                                interpolation=cv2.INTER_AREA)

        self.pts: Dict[int, List[float]] = {}         # frame-px (float)
        self.excluded: set[int] = set()
        self.dragging: int | None = None
        self.sel: int = min(POINTS)                   # keyboard-selected pt
        self.show_help = True
        self._live: CalibrationIntegration | None = None
        self._live_err: float = -1.0
        self._dirty = True
        self._init_positions()

        self.win = f"drag-calib [{name}]"
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self._on_mouse)

    # ---- positions -------------------------------------------------------
    def _init_positions(self) -> None:
        saved = CALIB_DIR / f"{self.name}_points.json"
        if saved.exists():
            data = json.loads(saved.read_text())
            for k, v in data.get("points", {}).items():
                self.pts[int(k)] = [float(v[0]), float(v[1])]
            self.excluded = {int(n) for n in data.get("excluded", [])}
        guess = _load_guess(self.name)
        for n, ((xc, yc), _d) in POINTS.items():
            if n in self.pts:
                continue
            if guess is not None:
                px, py = guess.court_to_image(xc, yc)
            else:
                px, py = self.fw / 2.0, self.fh / 2.0
            # clamp into the frame so EVERY dot is on-canvas & grabbable
            self.pts[n] = [float(np.clip(px, 0, self.fw - 1)),
                           float(np.clip(py, 0, self.fh - 1))]

    def _reset(self) -> None:
        guess = _load_guess(self.name)
        if guess is None:
            print("no current calibration to reset to")
            return
        for n, ((xc, yc), _d) in POINTS.items():
            px, py = guess.court_to_image(xc, yc)
            self.pts[n] = [float(np.clip(px, 0, self.fw - 1)),
                           float(np.clip(py, 0, self.fh - 1))]
        self._dirty = True
        print("dots reset to current-H projection")

    def _recompute_live(self) -> None:
        """Solve a live homography from the currently-included dots so the
        green court overlay tracks the dots in real time."""
        self._dirty = False
        inc = [n for n in sorted(self.pts) if n not in self.excluded]
        if len(inc) < 4:
            self._live, self._live_err = None, -1.0
            return
        court_pts = [POINTS[n][0] for n in inc]
        image_pts = [(self.pts[n][0], self.pts[n][1]) for n in inc]
        c = CalibrationIntegration()
        if not c.compute_homography(court_pts, image_pts):
            self._live, self._live_err = None, -1.0
            return
        errs = [((c.court_to_image(xc, yc)[0] - ix) ** 2 +
                 (c.court_to_image(xc, yc)[1] - iy) ** 2) ** 0.5
                for (xc, yc), (ix, iy) in zip(court_pts, image_pts)]
        self._live, self._live_err = c, float(np.mean(errs))

    # ---- coordinate maps -------------------------------------------------
    def _f2panel(self, x: float, y: float) -> Tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def _panel2f(self, x: int, y: int) -> Tuple[float, float]:
        return x / self.scale, y / self.scale

    def _nearest(self, fx: float, fy: float) -> int | None:
        best, bd = None, 1e9
        for n, (px, py) in self.pts.items():
            d = (px - fx) ** 2 + (py - fy) ** 2
            if d < bd:
                best, bd = n, d
        # 22 display-px grab radius -> frame-px
        return best if bd ** 0.5 <= 22 / self.scale else None

    # ---- mouse -----------------------------------------------------------
    def _on_mouse(self, ev: int, x: int, y: int, flags: int, _p) -> None:
        panel_w = int(round(self.fw * self.scale))
        if x > panel_w:                       # clicks on court panel ignored
            return
        fx, fy = self._panel2f(x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.dragging = self._nearest(fx, fy)
            if self.dragging is not None:
                self.sel = self.dragging
        elif ev == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            self.pts[self.dragging] = [
                float(np.clip(fx, 0, self.fw - 1)),
                float(np.clip(fy, 0, self.fh - 1))]
            self._dirty = True
        elif ev == cv2.EVENT_LBUTTONUP:
            self.dragging = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            n = self._nearest(fx, fy)
            if n is not None:
                self.excluded.symmetric_difference_update({n})
                self._dirty = True

    # ---- render ----------------------------------------------------------
    def _loupe(self, panel: np.ndarray, n: int) -> None:
        src = 90
        cx, cy = (int(self.pts[n][0]), int(self.pts[n][1]))
        x0, y0 = max(0, cx - src), max(0, cy - src)
        x1, y1 = min(self.fw, cx + src), min(self.fh, cy + src)
        crop = self.frame[y0:y1, x0:x1]
        if crop.size == 0:
            return
        zoom = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_NEAREST)
        rx = int((cx - x0) / max(1, x1 - x0) * 300)
        ry = int((cy - y0) / max(1, y1 - y0) * 300)
        cv2.drawMarker(zoom, (rx, ry), (0, 255, 255),
                       cv2.MARKER_CROSS, 26, 1)
        cv2.rectangle(zoom, (0, 0), (299, 299), (0, 255, 255), 2)
        cv2.putText(zoom, f"#{n}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        ph, pw = panel.shape[:2]
        panel[ph - 300:ph, pw - 300:pw] = zoom

    def _frame_panel(self) -> np.ndarray:
        disp = cv2.resize(self.frame,
                          (int(self.fw * self.scale), self.disp_h),
                          interpolation=cv2.INTER_AREA)
        if self._dirty:
            self._recompute_live()
        if self._live is not None:
            for poly in _court_polylines():
                ip = self._live.court_to_image_batch(poly)
                sp = [self._f2panel(x, y) for x, y in ip]
                for a, b in zip(sp, sp[1:]):
                    cv2.line(disp, a, b, GREEN, 2, cv2.LINE_AA)
        for n, (fx, fy) in self.pts.items():
            px, py = self._f2panel(fx, fy)
            col = GREY if n in self.excluded else RED
            r = 8 if n == self.dragging else 6
            cv2.circle(disp, (px, py), r, col, -1)
            cv2.circle(disp, (px, py), r, (255, 255, 255), 1, cv2.LINE_AA)
            t = (px + 9, py - 9)
            cv2.putText(disp, str(n), t, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, str(n), t, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        col, 1, cv2.LINE_AA)
        if self.sel in self.pts:                       # selection highlight
            sx, sy = self._f2panel(*self.pts[self.sel])
            cv2.circle(disp, (sx, sy), 13, CYAN, 2, cv2.LINE_AA)
        focus = self.dragging if self.dragging is not None else self.sel
        if focus in self.pts:
            self._loupe(disp, focus)
        used = len(self.pts) - len(self.excluded)
        err = (f"live reproj mean={self._live_err:.1f}px"
               if self._live is not None else "live overlay: need >=4 pts")
        seldesc = POINTS[self.sel][1] if self.sel in POINTS else "-"
        bar = [f"{self.name}  drag dots onto the lines  "
               f"(GREEN = live overlay)",
               f"included={used}  excluded={sorted(self.excluded)}  {err}",
               f"SELECTED #{self.sel} = {seldesc}",
               "[ ]=prev/next  i k j l=nudge1  IKJL=nudge10  x=excl  "
               "g=gather  s=save  r=reset  o=overlay  h=help  q=quit"]
        if self.show_help:
            for i, s in enumerate(bar):
                o = (14, 28 + i * 26)
                cv2.putText(disp, s, o, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(disp, s, o, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
        return disp

    def _canvas(self) -> np.ndarray:
        a = self._frame_panel()
        div = np.full((self.disp_h, 6, 3), 60, dtype=np.uint8)
        return np.hstack([a, div, self.court])

    # ---- persistence / solve --------------------------------------------
    def _save_points(self) -> None:
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        out = {"points": {str(n): [round(v[0], 1), round(v[1], 1)]
                          for n, v in sorted(self.pts.items())},
               "excluded": sorted(self.excluded)}
        (CALIB_DIR / f"{self.name}_points.json").write_text(
            json.dumps(out, indent=2))
        (CALIB_DIR / f"{self.name}_points_legend.txt").write_text(
            _legend_text(self.name))

    def _solve(self) -> None:
        self._save_points()
        confirmed = {n: (int(round(v[0])), int(round(v[1])))
                     for n, v in self.pts.items() if n not in self.excluded}
        if len(confirmed) < 4:
            print(f"need >=4 included points, have {len(confirmed)}")
            return
        try:
            _solve(self.name, self.frame_path, self.frame, confirmed)
        except SystemExit as e:
            print(f"solve failed: {e}")

    def _open_overlay(self) -> None:
        ov = CALIB_DIR / f"{self.name}_overlay.jpg"
        if not ov.exists():
            print("no overlay yet — press s to solve first")
            return
        img = cv2.imread(str(ov))
        s = self.disp_h / img.shape[0]
        cv2.imshow(f"overlay [{self.name}]",
                   cv2.resize(img, (int(img.shape[1] * s), self.disp_h)))

    # ---- loop ------------------------------------------------------------
    def run(self) -> None:
        print(f"[{self.name}] drag dots; s=solve, q=quit")
        while True:
            cv2.imshow(self.win, self._canvas())
            k = cv2.waitKey(16) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                self._solve()
                self._open_overlay()
            elif k == ord("r"):
                self._reset()
            elif k == ord("o"):
                self._open_overlay()
            elif k == ord("h"):
                self.show_help = not self.show_help
            elif k in (ord("["), ord("]")):
                order = sorted(self.pts)
                i = order.index(self.sel) if self.sel in order else 0
                self.sel = order[(i + (1 if k == ord("]") else -1))
                              % len(order)]
            elif k == ord("x"):
                self.excluded.symmetric_difference_update({self.sel})
                self._dirty = True
            elif k == ord("g"):                        # gather strays in-frame
                for n, (px, py) in self.pts.items():
                    self.pts[n] = [float(np.clip(px, 0, self.fw - 1)),
                                   float(np.clip(py, 0, self.fh - 1))]
                self._dirty = True
            elif k in (ord("i"), ord("k"), ord("j"), ord("l"),
                       ord("I"), ord("K"), ord("J"), ord("L")):
                step = 10.0 if k < ord("a") else 1.0    # capital = coarse
                dx = {ord("j"): -1, ord("l"): 1, ord("J"): -1,
                      ord("L"): 1}.get(k, 0) * step
                dy = {ord("i"): -1, ord("k"): 1, ord("I"): -1,
                      ord("K"): 1}.get(k, 0) * step
                px, py = self.pts[self.sel]
                self.pts[self.sel] = [
                    float(np.clip(px + dx, 0, self.fw - 1)),
                    float(np.clip(py + dy, 0, self.fh - 1))]
                self._dirty = True
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                break
        self._save_points()
        cv2.destroyAllWindows()
        print(f"saved {CALIB_DIR}/{self.name}_points.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="stem, e.g. FL_cal / NL_cal")
    p.add_argument("--frame", type=Path,
                   help="override (default demo/frames/<name>.jpg)")
    a = p.parse_args(argv)
    fp = a.frame or (FRAME_DIR / f"{a.name}.jpg")
    DragCalib(a.name, fp).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
