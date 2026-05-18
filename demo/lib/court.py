"""
Facility court geometry + top-down diagram renderer.

Geometry is the EXACT court CAD (uball_court_mapping/court_2.dxf,
units = cm), with the court-rectangle corner shifted to origin (0,0).
This replaces the previous NBA-regulation assumption, which did not
match this facility's (shorter, narrower) court and caused every
camera's key / FT-circle / 3pt arc to mis-align.

Court coordinate system:
  - X = length, 0 .. 2143.7 cm (baseline to baseline)
  - Y = width,  0 .. 1426.4 cm (sideline to sideline)
  - origin (0,0) = one baseline/sideline corner

Named landmarks below are the points picked in the frame during
calibration. Pick crisp painted-line intersections, well spread.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# --- facility court constants (cm), from court_2.dxf ---
COURT_LENGTH_CM: float = 2143.7          # baseline to baseline (X)
COURT_WIDTH_CM: float = 1426.4           # sideline to sideline (Y)
CENTER_X: float = COURT_LENGTH_CM / 2.0  # 1071.85
CENTER_Y: float = COURT_WIDTH_CM / 2.0   # 713.20
CENTER_CIRCLE_R_CM: float = 182.9        # outer center circle
CENTER_INNER_R_CM: float = 61.0          # inner center circle
LANE_WIDTH_CM: float = 388.6             # painted lane width
LANE_HALF: float = LANE_WIDTH_CM / 2.0   # 194.3
FT_DISTANCE_CM: float = 594.3            # baseline to free-throw line
FT_CIRCLE_R_CM: float = 182.9            # free-throw circle radius (DXF)
HOOP_FROM_BASELINE_CM: float = 107.7     # hoop / 3pt-arc center inset
THREE_R_CM: float = 670.6                # 3pt arc radius (about hoop)

LANDMARKS_CM: Dict[str, Tuple[float, float]] = {
    # Left-baseline end (x≈0)
    "L_baseline_top":      (0.0, 0.0),
    "L_baseline_bot":      (0.0, COURT_WIDTH_CM),
    "L_lane_base_top":     (0.0, CENTER_Y - LANE_HALF),            # (0, 518.9)
    "L_lane_base_bot":     (0.0, CENTER_Y + LANE_HALF),            # (0, 907.5)
    "L_ft_top":            (FT_DISTANCE_CM, CENTER_Y - LANE_HALF),  # (594.3,518.9)
    "L_ft_bot":            (FT_DISTANCE_CM, CENTER_Y + LANE_HALF),  # (594.3,907.5)
    "L_ft_center":         (FT_DISTANCE_CM, CENTER_Y),             # (594.3,713.2)
    # Center
    "center":              (CENTER_X, CENTER_Y),
    "center_top":          (CENTER_X, 0.0),
    "center_bot":          (CENTER_X, COURT_WIDTH_CM),
    "center_circle_top":   (CENTER_X, CENTER_Y - CENTER_CIRCLE_R_CM),
    "center_circle_bot":   (CENTER_X, CENTER_Y + CENTER_CIRCLE_R_CM),
    # Right-baseline end (x≈L)
    "R_baseline_top":      (COURT_LENGTH_CM, 0.0),
    "R_baseline_bot":      (COURT_LENGTH_CM, COURT_WIDTH_CM),
    "R_lane_base_top":     (COURT_LENGTH_CM, CENTER_Y - LANE_HALF),
    "R_lane_base_bot":     (COURT_LENGTH_CM, CENTER_Y + LANE_HALF),
    "R_ft_top":            (COURT_LENGTH_CM - FT_DISTANCE_CM, CENTER_Y - LANE_HALF),
    "R_ft_bot":            (COURT_LENGTH_CM - FT_DISTANCE_CM, CENTER_Y + LANE_HALF),
    "R_ft_center":         (COURT_LENGTH_CM - FT_DISTANCE_CM, CENTER_Y),
}


def cm_to_canvas(x_cm: float, y_cm: float, scale: float, pad: int) -> Tuple[int, int]:
    """Court cm -> top-down canvas pixel (X right, Y down)."""
    return int(round(x_cm * scale)) + pad, int(round(y_cm * scale)) + pad


def _arc_pts(cx: float, cy: float, r: float, a0_deg: float, a1_deg: float,
             n: int = 64):
    """Sample an arc from a0->a1 degrees (CCW), screen Y-down convention."""
    if a1_deg <= a0_deg:
        a1_deg += 360.0
    ts = np.linspace(np.radians(a0_deg), np.radians(a1_deg), n)
    return [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in ts]


def draw_topdown_court(scale: float = 0.30, pad: int = 40) -> np.ndarray:
    """Render the facility court top-down (matches court_2.dxf)."""
    w = int(round(COURT_LENGTH_CM * scale)) + 2 * pad
    h = int(round(COURT_WIDTH_CM * scale)) + 2 * pad
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    line = (235, 235, 235)
    th = 2

    def P(x_cm: float, y_cm: float) -> Tuple[int, int]:
        return cm_to_canvas(x_cm, y_cm, scale, pad)

    # Outer boundary + center line
    cv2.rectangle(canvas, P(0, 0), P(COURT_LENGTH_CM, COURT_WIDTH_CM), line, th)
    cv2.line(canvas, P(CENTER_X, 0), P(CENTER_X, COURT_WIDTH_CM), line, th)
    # Center circles
    cv2.circle(canvas, P(CENTER_X, CENTER_Y),
               int(CENTER_CIRCLE_R_CM * scale), line, th)
    cv2.circle(canvas, P(CENTER_X, CENTER_Y),
               int(CENTER_INNER_R_CM * scale), line, th)
    # Both ends: key + FT circle + 3pt arc + hoop
    for base_x, sign in ((0.0, 1.0), (COURT_LENGTH_CM, -1.0)):
        ft_x = base_x + sign * FT_DISTANCE_CM
        cv2.rectangle(canvas, P(base_x, CENTER_Y - LANE_HALF),
                      P(ft_x, CENTER_Y + LANE_HALF), line, th)
        cv2.circle(canvas, P(ft_x, CENTER_Y),
                   int(FT_CIRCLE_R_CM * scale), line, th)
        hoop_x = base_x + sign * HOOP_FROM_BASELINE_CM
        a0, a1 = (270, 90) if sign > 0 else (90, 270)
        arc = np.array([P(x, y) for x, y in
                        _arc_pts(hoop_x, CENTER_Y, THREE_R_CM, a0, a1)],
                       np.int32)
        cv2.polylines(canvas, [arc], False, line, th)
        cv2.circle(canvas, P(hoop_x, CENTER_Y), max(3, int(23 * scale)),
                   (0, 140, 255), th)
    return canvas
