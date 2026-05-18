# 07 — Operator Calibration Runbook (FL + NL)

You run this **once** on a machine with a GUI (your Mac). It produces the two
per-camera homographies the dual-camera fusion needs. The headless agent built
everything else and **cannot click** — this is the one human-in-the-loop step.
Cameras are bolted in place, so this is done **once and reused for every game**.

Everything is already staged: calibration frames pulled, tooling tested, and a
no-clicking real-footage validator ready for the moment you finish.

---

## 0. The single most important rule: ONE shared court convention

Both cameras must click the **same physical points** in the **same coordinate
convention**, or the two homographies will not share a space and fusion will be
nonsense. The convention (see `demo/frames/court_convention_schematic.jpg`):

| Axis | 0 | max | Pinned to (permanent feature) |
|---|---|---|---|
| **X** (length) | 0 | 2865 cm | **X=0 = the AMG-branded key baseline** (the hoop the **NL** camera is mounted directly above). X=2865 = the opposite baseline (behind FL, the spectator-bench end). |
| **Y** (width) | 0 | 1524 cm | **Y=0 = the scoreboard-side sideline** (digital scoreboard / Jordan-#23 poster wall). Y=1524 = the Iverson-#3-poster sideline. |

Landmark name decoder (`demo/lib/court.py` `LANDMARKS_CM`):

- `L_*` = the **AMG-key end** (X≈0). `R_*` = the far/bench end (X≈2865).
- `*_top` = the **Y=0 scoreboard side**. `*_bot` = the Y=1524 Iverson side.
- `L_lane_base_*` = AMG-key corners **on the baseline**; `L_ft_*` = AMG-key
  corners **on the free-throw line**; `L_ft_center` = middle of that FT line.
- `center*` = the COURTSIDE center-logo circle.

Keep the schematic open while clicking. Same physical point, same name, both cameras.

---

## 1. Frames are already pulled (t=1800 — cleanest)

`demo/frames/FL_cal.jpg` (2 players only, all lines crisp; far baseline
permanently bench-occluded — skip `R_*` for FL) and `demo/frames/NL_cal.jpg`
(near-empty; AMG key huge and pristine). Re-pull another time only if you want:
`python demo/extract_frame.py --angle FL --t <sec> --out demo/frames/FL_cal.jpg`.

> Both cameras are **fisheye** (bowed lines, vignette corners). For the demo we
> use central-region points and accept mild edge error (06 §4 opt 2). So:
> **prefer landmarks nearer the image centre; click the painted-line
> intersection precisely (zoom the window).**

---

## 2. Calibrate each camera (the clicking)

```bash
# FL — click these when prompted, SKIP (press s) any you can't see clearly:
#   L_lane_base_top, L_lane_base_bot, L_ft_top, L_ft_bot, L_ft_center,
#   center, center_circle_top, center_circle_bot, center_top, center_bot
#   (skip all R_* and L_baseline_* — bench-occluded in FL)
python demo/calibrate_homography.py --frame demo/frames/FL_cal.jpg --interactive

# NL — click the SAME shared landmarks (AMG key + center are the overlap that
#   ties the two cameras together), plus any others clearly visible:
#   L_baseline_top, L_baseline_bot, L_lane_base_top, L_lane_base_bot,
#   L_ft_top, L_ft_bot, L_ft_center, center, center_circle_top, center_circle_bot
python demo/calibrate_homography.py --frame demo/frames/NL_cal.jpg --interactive
```

Clicker keys: click in the prompted order · `s` skip landmark · `u` undo ·
`q` done. Need ≥4 (aim 6–8 well-spread). It writes per camera:
`demo/calibration/FL_cal_{H.npy,calibration.json,corr.json,overlay.jpg,topdown.jpg}`
(and `NL_cal_*`).

**Check before moving on:** open `FL_cal_overlay.jpg` / `NL_cal_overlay.jpg` —
the green reprojected court lines must sit on the real painted lines across the
**whole visible floor**, not just near your clicks. If they drift, re-run and
pick points spread wider / more central. Reprojection error prints to console.

**Shared-overlap sanity (the dual-camera-specific check):** the AMG-key
corners + `center` were clicked in *both* cameras. They are the same physical
points, so both `*_calibration.json` map them to the same court cm — that is
what makes the two views align.

---

## 3. Bind the two into one reusable manifest

```bash
python demo/calibrate_dual.py --game c2a354fe \
    --fl demo/calibration/FL_cal_calibration.json \
    --nl demo/calibration/NL_cal_calibration.json
# -> demo/calibration/c2a354fe_dual.json   (reuse forever for this rig)
```

---

## 4. Validate fusion on REAL footage (no clicking)

```bash
python demo/validate_dual.py --game c2a354fe --t 1400 \
    --fl-calib demo/calibration/FL_cal_calibration.json \
    --nl-calib demo/calibration/NL_cal_calibration.json
# -> demo/validation/c2a354fe_t1400_dualcheck.jpg
```

This pulls a live FL+NL frame-pair, runs YOLO, fuses, and renders
`[ FL+boxes | NL+boxes | top-down court+dots ]`. **Pass criteria:**

1. Dots land on the court at sensible positions (a player under the AMG hoop
   appears at the AMG end of the diagram, etc.).
2. A player standing in the **overlap** (mid-court, seen by both cameras)
   shows as **one green dot**, not a blue + an orange a long way apart.
   Green-but-slightly-off is fine for the demo; two far-apart dots = the two
   calibrations disagree → redo the camera whose `overlay.jpg` looked worse.

Try a couple of timestamps (`--t 900`, `--t 2700`). When overlap players are
reliably single green dots, calibration is signed off and we proceed to the
Step-2 detect+track loop on the full clip.

---

## If something's off

| Symptom | Likely cause | Fix |
|---|---|---|
| `overlay.jpg` lines fan off the court | points too clustered / mis-clicked | re-run, spread points, zoom to click line intersections precisely |
| Overlap player = far-apart blue & orange | the two cameras used different physical points / convention | re-check §0; the AMG-key corners + `center` must be the *same* physical spots in both |
| Dots mirrored / on wrong end | `*_top`/`*_bot` or `L`/`R` swapped vs §0 | re-click following the schematic exactly |
| FL far third empty of dots | expected — FL can't see it (bench); NL covers that end | not an error |
