"""
NBA full-court geometry + top-down diagram renderer.

Court coordinate system (matches vendored CalibrationIntegration):
  - X = length, 0 .. 2865 cm (baseline to baseline)
  - Y = width,  0 .. 1524 cm (sideline to sideline)
  - origin (0,0) = one baseline/sideline corner

Named landmarks below are the points you pick in the frame during
calibration (03_DEMO_BUILD_PLAN.md Step 1). Pick landmarks that are
crisp painted-line intersections and well spread across the court.

Metric exactness is NOT required for the uniform-dot demo — only that
image picks and these court coords refer to the SAME physical point.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# --- NBA full-court constants (cm) ---
COURT_LENGTH_CM: float = 2865.0          # baseline to baseline (X)
COURT_WIDTH_CM: float = 1524.0           # sideline to sideline (Y)
CENTER_X: float = COURT_LENGTH_CM / 2.0  # 1432.5
CENTER_Y: float = COURT_WIDTH_CM / 2.0   # 762.0
CENTER_CIRCLE_R_CM: float = 183.0        # 6 ft radius
LANE_WIDTH_CM: float = 488.0             # 16 ft painted lane width
FT_DISTANCE_CM: float = 579.0            # baseline to free-throw line (19 ft)
HOOP_FROM_BASELINE_CM: float = 160.0     # backboard/hoop center inset (~5.25 ft)
THREE_R_CM: float = 723.9                # 3pt arc radius (23.75 ft)

# Named court landmarks in (x_cm, y_cm). "L" baseline = x≈0, "R" baseline = x≈2865.
# Lane spans y in [CENTER_Y - LANE_WIDTH/2, CENTER_Y + LANE_WIDTH/2] = [518, 1006].
LANE_HALF = LANE_WIDTH_CM / 2.0
LANDMARKS_CM: Dict[str, Tuple[float, float]] = {
    # Left-baseline end
    "L_baseline_top":      (0.0, 0.0),
    "L_baseline_bot":      (0.0, COURT_WIDTH_CM),
    "L_lane_base_top":     (0.0, CENTER_Y - LANE_HALF),     # (0, 518)
    "L_lane_base_bot":     (0.0, CENTER_Y + LANE_HALF),     # (0, 1006)
    "L_ft_top":            (FT_DISTANCE_CM, CENTER_Y - LANE_HALF),  # (579, 518)
    "L_ft_bot":            (FT_DISTANCE_CM, CENTER_Y + LANE_HALF),  # (579, 1006)
    "L_ft_center":         (FT_DISTANCE_CM, CENTER_Y),             # (579, 762)
    # Center
    "center":              (CENTER_X, CENTER_Y),
    "center_top":          (CENTER_X, 0.0),
    "center_bot":          (CENTER_X, COURT_WIDTH_CM),
    "center_circle_top":   (CENTER_X, CENTER_Y - CENTER_CIRCLE_R_CM),
    "center_circle_bot":   (CENTER_X, CENTER_Y + CENTER_CIRCLE_R_CM),
    # Right-baseline end
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


def draw_topdown_court(scale: float = 0.30, pad: int = 40) -> np.ndarray:
    """
    Render a clean top-down NBA half-and-full court diagram.

    scale = canvas px per cm (0.30 -> ~860x457 court + padding).
    Returns a BGR canvas. Players (Step 4) get drawn on a copy of this.
    """
    w = int(round(COURT_LENGTH_CM * scale)) + 2 * pad
    h = int(round(COURT_WIDTH_CM * scale)) + 2 * pad
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)  # dark wood-ish bg

    line = (235, 235, 235)
    th = 2

    def P(x_cm: float, y_cm: float) -> Tuple[int, int]:
        return cm_to_canvas(x_cm, y_cm, scale, pad)

    # Outer boundary
    cv2.rectangle(canvas, P(0, 0), P(COURT_LENGTH_CM, COURT_WIDTH_CM), line, th)
    # Center line + circle
    cv2.line(canvas, P(CENTER_X, 0), P(CENTER_X, COURT_WIDTH_CM), line, th)
    cv2.circle(canvas, P(CENTER_X, CENTER_Y),
               int(CENTER_CIRCLE_R_CM * scale), line, th)
    # Both keys + free-throw circles + hoops
    for base_x, sign in ((0.0, 1.0), (COURT_LENGTH_CM, -1.0)):
        ft_x = base_x + sign * FT_DISTANCE_CM
        cv2.rectangle(canvas,
                      P(base_x, CENTER_Y - LANE_HALF),
                      P(ft_x, CENTER_Y + LANE_HALF), line, th)
        cv2.circle(canvas, P(ft_x, CENTER_Y),
                   int(LANE_HALF * scale), line, th)
        hoop_x = base_x + sign * HOOP_FROM_BASELINE_CM
        cv2.circle(canvas, P(hoop_x, CENTER_Y), max(3, int(23 * scale)),
                   (0, 140, 255), th)
    return canvas
