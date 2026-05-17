# 03 — Demo Build Plan: Synced Video + Court Map

Goal: one self-contained output mp4 — left panel = processed game video with shot events overlaid, right panel = top-down 2D court with players as **uniform-colored dots** (every player the same color), perfectly time-synced.

> **Scope decision (2026-05-17):** team-color split is **deferred**. For this demo every player is the *same* color dot — this is enough to show players moving on the court side-by-side with the real video. Jersey-color → team classification comes later, after a dedicated color-training pass. This removes the previously #1 demo risk from the critical path.

This is executable by a fresh Claude Code session. Each step says what to reuse vs. build.

---

## Pre-reqs

- A game with full-court camera coverage in S3 (the wide/far angle — confirm which). The validated game `c2a354fe` (S3: `court-a/2026-03-19/c2a354fe-eb34-4980-af00/`) is a safe default since we already have its fused shot output.
- Its fused `detection_results.json` (shot events) — from the Uball_dual_angle_fusion run, or regenerate (see `01_FUSION_LOGIC.md`).
- Python env with: `ultralytics`, `opencv-python`, `numpy`, `boto3`, the v16 far weights if re-running detection (see `04_REFERENCES.md` for S3 path).

---

## Step 1 — Establish court homography (once per camera)

**Reuse:** `BasketTracking-1/rectify_court.py:96–209` (auto court-corner detect → `getPerspectiveTransform`). **Fallback:** 4 manual court-corner clicks like `uball_court_mapping/app/services/calibration_integration.py` (`cv2.findHomography`).

- Pick one representative frame from the wide-angle video.
- Get 4+ court-landmark correspondences (corners, center circle, free-throw line intersections) between image pixels and real court coords (a standard court is ~28 m × 15 m FIBA, or use the DXF in `uball_court_mapping` = 2460×1730 cm).
- Compute + **cache** the 3×3 homography matrix `H` to a file (`demo/calibration/<game>_H.npy`).
- Sanity check: project the 4 corners back, overlay on the frame, confirm they land on the real court lines.

**Output:** `demo/calibration/<game>_H.npy` + a sanity-overlay JPG.

---

## Step 2 — Detect + track players, project to court

**Reuse:** `uball_court_mapping/app/services/player_detector.py` (YOLOv11 person) + `bytetrack_tracker.py` (ByteTrack). SAM2 mask refinement is optional — skip for demo speed unless dots jitter.

For each frame:
1. YOLOv11 → person bboxes
2. ByteTrack → stable track IDs
3. For each track: take the **foot point** = bbox bottom-center `(x1+x2)/2, y2`
4. Project foot point through `H` → court coordinates `(cx, cy)`

**Build-new (small):** a thin loop that reads the video frame-by-frame and calls the above, accumulating `{frame_idx: [{track_id, court_xy, bbox}]}`.

**Output:** `demo/tracks/<game>_tracks.json` (per-frame player court positions + track IDs).

---

## Step 3 — Player color (DEFERRED — uniform color for this demo)

**For this demo: skip team classification entirely.** Every player dot is rendered in **one fixed color** (e.g. white or cyan). No HSV sampling, no torso region, no clustering. This is a deliberate scope cut — see the scope decision at the top.

- Implementation: a single constant `PLAYER_DOT_COLOR` used for all tracks in Step 4.
- Keep the per-track `track_id` so dots are still individually trackable (and so a later pass can attach a team label without re-running detection).

**Output:** `<game>_tracks.json` is used as-is from Step 2 — no team field.

**Later (post-training, NOT this demo):** re-enable jersey-color → team classification by reusing `BasketTracking-1/player_detection.py:15–19` (HSV ranges) + classify logic `:73–188`, with auto-cluster (k-means k=2 on torso HSV across ~50 frames) + majority-vote per track + an `unknown` class. Adds a `team: "A"|"B"|"unknown"` field; Step 4 then maps team → color instead of the constant. The user will supply trained team colors at that point.

---

## Step 4 — Render the synced side-by-side

**Reuse:** `uball_court_mapping/app/services/video_stitcher.py` — `create_stitched_frame()` (:133), the dot renderer (:204–205), and `uwb_to_vertical_screen()` for the court-canvas transform. **Simplify** the color source: replace `persistent_id_mapper.py`'s per-tag color lookup with the single constant `PLAYER_DOT_COLOR`. (This is *less* work than the original per-team swap — the mapper can be bypassed entirely.)

Per frame:
- Left panel: the game video frame. Overlay any shot event active at this timestamp (Step 5).
- Right panel: a clean top-down court diagram (draw court lines once as a static background; reuse the DXF geometry from `uball_court_mapping/app/services/dxf_parser.py` or just hardcode a FIBA court). For each tracked player: `cv2.circle(court_canvas, project(court_xy), R, PLAYER_DOT_COLOR, -1)` + contrasting outline (same color for every player). Optionally a short fading trail (last ~15 positions).
- Concatenate panels horizontally, write to `cv2.VideoWriter` at the source FPS.

**Output:** `demo/<game>_demo.mp4`.

---

## Step 5 — Overlay shot events from the fused detection

**Reuse:** the fused `detection_results.json` schema (`01_FUSION_LOGIC.md` §4).

- Load `shots[]`. For each shot with `timestamp_seconds = t` and `outcome`:
  - When the video reaches `t ± 1.5 s`, flash a banner on the left panel: e.g. green "✓ MADE (conf 0.87)" or red "✗ MISS", optionally a small marker on the court at the shooting hoop.
- Keep it tasteful — this is the client "wow" moment proving detection + spatial understanding together.

---

## Step 6 — Package the deliverable

- One mp4 (`demo/<game>_demo.mp4`), ~the length of the clip you process. For a client demo, **process a 3–5 minute highlight slice**, not the full 56-min game (faster to produce, easier to watch).
- A 1-paragraph `demo/README.md` for the client: what they're looking at, the headline accuracy numbers (100% detection recall on c2a354fe — from `README.md`).

---

## Effort + sequencing

| Step | Reuse vs build | Est. effort |
|---|---|---|
| 1 Homography | reuse `BasketTracking-1` (+ manual fallback) | 0.5 day |
| 2 Detect+track+project | reuse `uball_court_mapping` services | 0.5 day |
| 3 Player color | **deferred — uniform color, ~0** | ~0 (one constant) |
| 4 Side-by-side render | reuse `video_stitcher.py`, bypass color mapper | 1 day |
| 5 Shot-event overlay | new, small | 0.5 day |
| 6 Package | trivial | 0.25 day |

**Total ≈ 2.5–3 days** for a polished client demo, mostly reuse + glue. Dropping team classification removes the previously dominant cost/risk (Step 3 jersey tuning), shaving ~1 day off the original 3.5–4 day estimate. Remaining iterative part is homography accuracy at the far court end (Step 1).

---

## Recommended demo game

Use **c2a354fe** (2026-03-19): we already have its fused shot output with **100% detection recall** (validated in `README.md`). The shot-event overlay will be accurate, making the spatial demo land with maximum credibility. Process a 3–5 min slice with lots of scoring action.

---

## Risks / gotchas

1. ~~Jersey color under gym lighting~~ — **no longer applicable to this demo** (uniform dots). Will become the dominant risk when team classification is re-enabled post-training; the auto-cluster + majority-vote + "unknown"-class mitigation is documented in Step 3's "Later" note for that phase.
2. **Homography accuracy at far court end** — now the #1 risk. Perspective error grows with distance from camera. Foot-point projection (not centroid) mitigates. Validate dots land plausibly at both ends.
3. **Players occluding each other** — ByteTrack handles most ID continuity; SAM2 mask refinement (available in `uball_court_mapping`) helps if dots merge/jump. Enable it if needed (slower). With uniform dots a brief ID swap is visually harmless (same color), so this is lower-stakes for this demo.
4. **Sync drift** — drive both panels off the *same* frame index / source FPS. Don't independently time the two streams.
5. **Don't over-scope** — first deliverable is one pre-rendered mp4 on one game, uniform dots. No team colors, no live processing, no UI, no multi-court. Land that, then iterate.
