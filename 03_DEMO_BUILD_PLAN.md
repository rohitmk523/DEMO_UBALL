# 03 — Demo Build Plan: Synced Video + Court Map

Goal: one self-contained output mp4 — left panel = processed game video with shot events overlaid, right panel = top-down 2D court with players as jersey-colored dots, perfectly time-synced.

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

## Step 3 — Jersey-color → team classification

**Reuse:** `BasketTracking-1/player_detection.py:15–19` (HSV ranges) + the classify logic in `:73–188`.

- For each player bbox, sample the **torso region** (upper-middle of the bbox, avoid head/legs/background).
- Convert to HSV, test against the two team color ranges.
- **Critical:** the hard-coded green/red ranges are placeholders. Either:
  - (a) auto-cluster: sample torso colors across ~50 frames, k-means(k=2) in HSV → two team centroids, OR
  - (b) config: expose `team_a_hsv`, `team_b_hsv` ranges, set them by eyeballing one frame.
- Assign each track a stable team via majority vote over its lifetime (a track shouldn't flip teams frame-to-frame).

**Output:** augment `<game>_tracks.json` with `team: "A"|"B"|"unknown"` per track.

---

## Step 4 — Render the synced side-by-side

**Reuse:** `uball_court_mapping/app/services/video_stitcher.py` — `create_stitched_frame()` (:133), the dot renderer (:204–205), and `uwb_to_vertical_screen()` for the court-canvas transform. **Swap** the color source: `persistent_id_mapper.py`'s per-tag color → per-**team** color from Step 3.

Per frame:
- Left panel: the game video frame. Overlay any shot event active at this timestamp (Step 5).
- Right panel: a clean top-down court diagram (draw court lines once as a static background; reuse the DXF geometry from `uball_court_mapping/app/services/dxf_parser.py` or just hardcode a FIBA court). For each tracked player: `cv2.circle(court_canvas, project(court_xy), R, TEAM_COLOR, -1)` + white outline. Optionally a short fading trail (last ~15 positions).
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
| 3 Jersey-color teams | reuse `BasketTracking-1` HSV + add auto-cluster | 1 day (tuning is the cost) |
| 4 Side-by-side render | reuse `video_stitcher.py`, swap color source | 1 day |
| 5 Shot-event overlay | new, small | 0.5 day |
| 6 Package | trivial | 0.25 day |

**Total ≈ 3.5–4 days** for a polished client demo, mostly reuse + glue. The risky/iterative part is Step 3 (jersey color robustness under varying lighting) — budget the most time there and test on 2+ games.

---

## Recommended demo game

Use **c2a354fe** (2026-03-19): we already have its fused shot output with **100% detection recall** (validated in `README.md`). The shot-event overlay will be accurate, making the spatial demo land with maximum credibility. Process a 3–5 min slice with lots of scoring action.

---

## Risks / gotchas

1. **Jersey color under gym lighting** — the single biggest risk. Two teams in similar colors, or shadows, break HSV classification. Mitigation: auto-cluster (Step 3a) + majority-vote per track + an "unknown" class rather than forcing a wrong team.
2. **Players occluding each other** — ByteTrack handles most ID continuity; SAM2 mask refinement (available in `uball_court_mapping`) helps if dots merge/jump. Enable it if needed (slower).
3. **Homography accuracy at far court end** — perspective error grows with distance from camera. Foot-point projection (not centroid) mitigates. Validate dots land plausibly at both ends.
4. **Sync drift** — drive both panels off the *same* frame index / source FPS. Don't independently time the two streams.
5. **Don't over-scope** — first deliverable is one pre-rendered mp4 on one game. No live processing, no UI, no multi-court. Land that, then iterate.
