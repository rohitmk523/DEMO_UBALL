# 05 — Step 1 Execution Notes: Court Homography

Status: **tooling complete + runnable; usable full-court H is BLOCKED on a clean calibration pass.** Honest result below.

---

## What was built (reusable, committed)

| Path | What it is |
|---|---|
| `demo/lib/calibration_integration.py` | **Vendored verbatim** from `uball_court_mapping` (04_REFERENCES verdict: COPY). The homography core: `compute_homography`, `court_to_image`, `image_to_court`, JSON save/load. cv2 4.12 clean. |
| `demo/lib/court.py` | NBA full-court geometry (2865×1524 cm), named landmarks dict, `draw_topdown_court()` diagram renderer (reused again in Step 4). |
| `demo/extract_frame.py` | Pulls one frame from the multi-GB S3 video **without downloading it** — `aws s3 presign` + `ffmpeg` HTTP range seek. Proven on FL/FR @1080p30. |
| `demo/calibrate_homography.py` | Step 1 driver. `--interactive` (cv2 click UI) **or** `--config corr.json` (headless/reproducible). Computes H via the vendored core, caches `*_H.npy` + `*_calibration.json`, renders `*_overlay.jpg` + `*_topdown.jpg` quality checks. |

Run end-to-end, no errors. Reprojection error on the 4 input points = 0.0 px (4 pts → exact floor-plane homography).

## Source video facts (c2a354fe)

- S3: `s3://uball-videos-production/court-a/2026-03-19/c2a354fe-eb34-4980-af00/2026-03-19_..._{FL,FR,NL,NR}.mp4`
- All 4 angles: **1920×1080, 29.97 fps, ~3597 s (~60 min)**, 3.7–4.9 GiB each.
- FL = far-left, FR = far-right. Both are **end-corner cameras shooting lengthwise** toward the far hoop.

## The blocking finding (this is the real Step 1 output)

The FL camera sits essentially **on the near baseline**. Consequence:

1. **Near key is off-frame** (camera is on top of it) — no near-court painted box.
2. **Far baseline + its corners are occluded** by the spectator bench/chairs along the far end.
3. **"Center circle" is a large stylized COURTSIDE logo**, not a crisp painted circle.
4. The **only crisp coplanar floor feature is the far painted red key** (lane rectangle + free-throw line) — and all its corners cluster into a small image slab (~x700–1180, y530–690).

A 4-point homography from that small cluster is mathematically exact at those 4 points but **extrapolates to garbage across the rest of the court** — see `demo/calibration/c2a354fe_FL_t1400_overlay.jpg` (projected court lines fan off the real court) and `_topdown.jpg` (only the key region warps plausibly; rest is empty/flipped). This **concretely confirms and quantifies the build plan's documented #1 risk** (far-end / under-conditioned homography) — for this footage it is a *blocker*, not a tuning nuisance, until a better calibration pass is done.

## Why it can't be finished headless

A good full-court H needs 6+ well-spread **coplanar floor** landmarks. The candidates (center-logo ellipse extremes, sideline points, 3-pt arc, far-baseline corners) are either occluded, stylized, or not safely readable by eyeballing a downscaled remote frame. Picking them needs a human eye on the full-res frame — exactly the interactive clicker, which needs a local GUI (cannot run in this headless agent).

## Recommended next actions (decision needed — see chat)

1. **Calibrate on an EMPTY-court frame.** H is camera-fixed, so use a pre-game / timeout / warm-up timestamp where painted lines aren't occluded by players. `python demo/extract_frame.py --angle FL --t <empty_t>` then `--interactive`. Biggest cheap win.
2. **Operator runs the interactive clicker** on that clean frame, picking the painted key + center-logo ellipse + both visible sideline/3-pt-arc points for spread. (Tool is ready.)
3. **Far-third of court is structurally weak from FL alone** (occluded far baseline). Options to decide: (a) accept lower dot accuracy there for the demo, (b) frame the demo around the well-covered ~⅔ court, or (c) fuse a near camera (NL/NR) for the far end — *out of current demo scope; flag if pursued*.
4. Re-run; iterate until `_overlay.jpg` lines sit on the painted court across the whole floor.

## Artifacts from this pass (committed as evidence)

`demo/frames/c2a354fe_{FL,FR}_t1400.jpg`, `demo/calibration/c2a354fe_FL_t1400_{H.npy,calibration.json,corr.json,overlay.jpg,topdown.jpg}`. The H here is a **placeholder** (key-only, not demo-usable) kept so the pipeline is exercised end-to-end and the refinement is a drop-in re-run.
