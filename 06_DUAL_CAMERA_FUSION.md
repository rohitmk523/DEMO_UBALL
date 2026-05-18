# 06 — Dual-Camera Court-Mapping Fusion (Design)

Decision (2026-05-18): the single FL camera cannot cover the full court (far baseline occluded, near key off-frame — see [`05_STEP1_NOTES.md`](05_STEP1_NOTES.md)). We fuse **FL + NL**: two fixed cameras at opposite ends, each projected into one shared court space. **Design only — no implementation until this is signed off.**

---

## 1. The core principle: fuse in court space, not image space

The two cameras are **never compared image-to-image**. Each has its own homography mapping its pixels into the **same** canonical court (NBA 2865×1524 cm, already in `demo/lib/court.py`):

```
FL image px ──H_FL──┐
                     ├──▶  ONE shared court space (cm)  ──▶  merge ──▶ 2D dot map
NL image px ──H_NL──┘
```

A player at court (2400, 600) cm is at (2400, 600) **regardless of which camera saw them or from what direction**. "Opposite-side cameras" only means H_FL and H_NL carry very different rotations — they still resolve into the same metric frame. This is why the mounting geometry is a non-issue once you stop thinking in pixels.

## 2. Why FL + NL are complementary (verified on c2a354fe frames)

| Camera | Mounted | Strong zone | Weak zone |
|---|---|---|---|
| **FL** | End A baseline, lengthwise | Near + mid court (center logo) | Far third — far baseline **occluded by spectator bench**; its own near key off-frame |
| **NL** | Above End B hoop (AMG key), lengthwise back | The AMG-key end + foreground — **huge detail right under it** | Far hoop (End A) tiny; **strong fisheye** at edges |

FL's weak third **is** NL's strongest zone and vice-versa. Each camera's homography is fit and trusted **only in its strong region**; the shared court space stitches them.

```
End A (FL here)                               End B (NL above this hoop)
FL  ███████████████ strong ░░░░░░░░ weak/occluded
NL  weak/tiny ░░░░░░░░ strong ███████████████
        └──── overlap band (≈ center third) ────┘   ← reconcile here
```

## 3. Calibrate ONCE per camera — never per game

Cameras are physically bolted. A homography depends only on (fixed) camera pose + the (fixed) court plane. So:

- One-time, offline: one **empty-court** frame per camera → `H_FL.npy`, `H_NL.npy` (+ NL lens-undistort map). Commit them.
- Every game thereafter: load the cached matrices, run detection. **Zero per-game calibration.**
- Re-calibrate only if a camera is physically moved. This is a headline operational win, not just an implementation detail.

The Step-1 tooling (`demo/calibrate_homography.py`, vendored `calibration_integration.py`) already persists matrices — dual-camera = two cached files instead of one.

## 4. Fisheye: the one real new problem

A plane homography assumes a pinhole camera. **NL/NR barrel-distort** (court boundary lines visibly bow; vignette corners — GoPro-class wide FOV). FL looks near-rectilinear. Options, in order of preference:

1. **One-time fisheye intrinsic calibration of NL** (`cv2.fisheye.calibrate` on a checkerboard or court-line fit) → undistort every NL frame before homography. Most correct; one-time cost; reusable forever (static lens).
2. Fit H_NL on **central-region** points only and accept growing edge error — acceptable for a uniform-dot demo, cheap, no checkerboard needed.
3. Hybrid: ship the demo with (2), upgrade to (1) if dot accuracy at NL edges is visibly off.

Recommendation: **start with (2)** for the demo timeline, design the code so undistortion is a pluggable pre-step to add (1) later.

## 5. Fusion algorithm (court space)

1. Detect + track players **independently per camera** (own tracker each).
2. Project each track's **foot point** through that camera's H (+ NL undistort) → court cm.
3. **Zone authority:** a seam (~center). FL-strong half from FL, NL-strong half from NL.
4. **Overlap band:** prefer the camera whose detection is physically closer (less foreshortening = more accurate); reconcile duplicates by court-space proximity.
5. **Cross-camera identity:** a player crossing the seam = two track IDs (one per camera) stitched by court-space proximity + temporal continuity. **Uniform dots make this forgiving** — a brief mismatch is visually invisible (every dot identical). The deferred-team-color decision pays off again here.

## 6. Reuse map — do NOT build the merge from scratch

`trackingStudio/` already implements dual-camera court-space fusion. Evaluated file:line (verdicts):

| Component | Verdict | Source | Notes |
|---|---|---|---|
| `CrossCameraMerger` class | **REF only** (was COPY) | `trackstudio/processors/cross_camera_merger.py:60–495` | Vendored at `demo/lib/cross_camera_merger.py` for provenance, **not on the demo path**. Implementation finding (encoded in `demo/tests/test_dual_fusion.py`): it only matches a detection against **prior-frame** global players; `_associate_players_across_cameras` merely *logs* candidate pairs, never links them. So two cameras seeing a player in the **same** frame each spawn a separate global and never reconcile. The `SpatialCrossCameraMerger` threshold fix (`demo/lib/spatial_merger.py`) does **not** cure this — it is structural. We therefore wrote our own per-frame court-space fusion. |
| **Per-frame court-space fusion** | **OURS (new, small)** | `demo/lib/dual_camera_fusion.py` | Project foot points → shared court CM (vendored `CalibrationIntegration`), greedy mutual nearest-neighbour FL↔NL within `max_player_distance_cm`, mean position; unmatched pass through. 5 synthetic tests green. No global-ID state machine needed — uniform dots make persistent IDs unnecessary. |
| BEV projection math | **ADAPT** | `video_processor.py:1013–1043` (`_transform_point_to_bev`) | We instead reuse our vendored `calibration_integration.court_to_image`/`image_to_court` for consistency with Step 1; take the merge/court-frame structure from here. |
| Homography setup | **SKIP** (have better) | `video_processor.py:719–742` | Per-session, not persisted. Our Step-1 `calibrate_homography.py` already persists `*_H.npy` — keep ours. |
| DeepSORT tracker | **REF / optional** | `processors/deepsort_tracker.py` | Gives appearance embeddings (helps cross-cam re-ID). But same-jersey players + uniform dots make appearance weak; **court-space proximity is the robust signal**. Default to `uball_court_mapping` ByteTrack (already chosen in [02](02_COURT_MAPPING.md)); use merger's spatial score, drop/down-weight appearance. |
| Fisheye undistortion | **IMPLEMENT** | — | Not in trackingStudio either (it assumes rectilinear input). New, NL-only, one-time. §4. |
| FastAPI layer | **SKIP** | `trackstudio/api/` | Trivial to drop; merger has zero framework coupling. |

Net new code is small: a fisheye pre-step + glue. Detection/tracking = `uball_court_mapping` (per [02](02_COURT_MAPPING.md)); calibration = vendored core (Step 1); cross-camera merge = `trackingStudio` `CrossCameraMerger`; render = `uball_court_mapping` `video_stitcher` (per [03](03_DEMO_BUILD_PLAN.md) Step 4).

## 7. Revised pipeline & effort delta vs single-camera

| Step | Single-cam | Dual-cam (this design) | Δ |
|---|---|---|---|
| 1 Homography | 1 H | **2 H** (FL, NL) + 1 NL undistort map, all calibrate-once | +0.5 d |
| 2 Detect+track+project | 1 stream | **2 streams**, each → court space | +0.5 d |
| 2b Cross-camera merge | — | reuse `CrossCameraMerger` (COPY) + tune | +0.5 d |
| 3 Player color | uniform (deferred) | unchanged | 0 |
| 4 Render side-by-side | as-is | court panel now fed by merged tracks | ~0 |
| 5 Shot overlay | as-is | unchanged | 0 |

Net ≈ **+1.5–2 days** over the ~2.5–3 d single-camera estimate → **~4–5 days**. Buys full-court coverage + the calibrate-once operational win.

## 8. Risks

| Risk | Mitigation |
|---|---|
| NL fisheye → H error at edges | §4: central-fit now, intrinsic undistort upgrade later |
| Two calibrations disagree in overlap | Validate: a player in both cameras must land at the same court (x,y); overlay both cameras' dots on a shared-court frame to check |
| Seam discontinuity (dot jump FL↔NL) | Blend transition band; place seam in cleanest overlap |
| Cross-camera ID swap | Court-space proximity + time; **uniform dots hide brief swaps** |
| trackingStudio court dims hard-coded (94×50 ft NBA) | Our `court.py` is the single court source of truth; feed merger our dims, don't use its hard-coded ones |

## 9. Open decisions (default = my pick unless you say otherwise)

- Near camera: **NL** (pairs with FL on the left optics). NR is near-identical; switch only if NL has a worse view at calibration time.
- Fisheye: **start central-fit (§4 opt 2)**, upgrade later.
- Seam: **center third overlap**, exact line tuned at integration.

## 10. Implementation status (2026-05-18)

Built + tested (`demo/lib/`, `demo/tests/test_dual_fusion.py` — 5 green):

- `cross_camera_merger.py` — vendored verbatim (provenance; REF only, see §6)
- `spatial_merger.py` — `SpatialCrossCameraMerger` subclass (REF; documents the no-appearance threshold issue)
- `dual_camera_fusion.py` — **the demo path**: own per-frame court-space fusion (foot-point project → greedy court-space merge)
- `calibrate_dual.py` — binds two per-camera Step-1 calibrations into one reusable manifest (`<game>_dual.json`)
- candidate calibration frames pulled: `demo/frames/{FL,NL}_cal_t2.jpg`

**Still blocked (unchanged):** real per-camera homographies need the operator interactive clicker on a local GUI (headless agent can't click). Everything above is runnable the moment `H_FL`/`H_NL` exist — the fusion math is already proven on synthetic identity calibrations. The video has no empty pre-game (play starts ~t=0), so calibrate on the least-occluded frame.

Next: revise [`03_DEMO_BUILD_PLAN.md`](03_DEMO_BUILD_PLAN.md) Steps 1–2 to the two-camera flow (done alongside this).
