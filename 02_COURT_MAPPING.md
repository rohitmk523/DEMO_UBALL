# 02 — Court Mapping & Player Tracking: Where the Code Is

The demo needs players rendered as **jersey-colored dots on a top-down 2D court**, synced to the game video. This doc maps every relevant codebase so the build reuses proven components.

---

## The three candidate repos

| Repo | What it is | Side-by-side render? | Jersey-color teams? | Homography | Verdict |
|---|---|---|---|---|---|
| **`uball_court_mapping/`** | Mature FastAPI system: YOLOv11 + ByteTrack + SAM2 → court projection → stitched dual-panel video | ✅ **yes** (`video_stitcher.py`) | ❌ no (colors by UWB tag_id) | manual point-click + DXF court geometry | **Best rendering scaffold** |
| **`BasketTracking-1/`** | Panorama-stitch + court rectification + player detection with HSV team colors → 2D minimap | ✅ yes (implied in main loop) | ✅ **yes** (HSV jersey classifier) | auto court-corner detection + `getPerspectiveTransform` | **Best jersey-color + homography source** |
| **`Uball_tracking/`** | Clean tracking-only package (YOLO+SAM2+ByteTrack), no court projection | ❌ no | ❌ no | ❌ none | tracking only — skip for court map |
| `trackingStudio/` | Dual-camera FastAPI, DeepSORT, BEV view, cross-camera merge | ✅ yes (BEV) | ❌ no | 4-corner per camera | alternative; weaker cross-cam re-ID |

**Strategy: combine the two best.** Use `uball_court_mapping`'s side-by-side stitching scaffold + `BasketTracking-1`'s jersey-color team classifier + homography. Neither alone is complete; together they cover the whole demo.

---

## A. `uball_court_mapping/` — the rendering scaffold (reuse this for the side-by-side)

FastAPI app; the pipeline orchestrator is `persistent_id_tracking.py`.

| File | What it does |
|---|---|
| `persistent_id_tracking.py` | **Main entrypoint.** Frame loop: PlayerDetector(YOLOv11) → SAM2 mask refine → ByteTrack → court projection → UWB associate → VideoStitcher |
| `app/services/player_detector.py` | YOLOv11 person detection (class 0) |
| `app/services/sam2_segmenter.py` | SAM2 mask refinement (optional) |
| `app/services/bytetrack_tracker.py` | ByteTrack with optional mask features |
| `app/services/calibration_integration.py` | **Homography**: `cv2.findHomography(court_pts, image_pts, cv2.RANSAC)` (5px). Saves `data/calibration/calibration.json` |
| `app/services/video_stitcher.py` | **The side-by-side renderer.** `create_stitched_frame()` :133. Player dot render at **:204** `cv2.circle(canvas,(cx,cy),DOT_RADIUS,color,-1)` + white outline :205. `uwb_to_vertical_screen()` rotates court 90° to a vertical canvas |
| `app/services/persistent_id_mapper.py` | maps track IDs → stable colors (currently per-tag, **swap this for per-team**) |
| `app/services/dxf_parser.py` | parses `court_2.dxf` → real court geometry (2460×1730 cm) |

**Existing outputs (proof it works):** `video_output/GX010018_..._bytetrack_sam_stitched.mp4` (~770 MB dual-panel files: video left, court+dots right).

**Gap:** colors dots by UWB `tag_id`, not jersey. The demo has no UWB tags → must replace the coloring source with jersey-color team classification (from repo B).

---

## B. `BasketTracking-1/` — jersey-color + homography (reuse this for team coloring)

This repo has the **actual jersey-color team classifier and a working 2D court projection**.

| File:line | What it does |
|---|---|
| `rectify_court.py:12–94` | `collage()` / `add_frame()` — SIFT+RANSAC panorama stitch (only needed if doing multi-frame court build; can skip for single fixed camera) |
| `rectify_court.py:96–157` | `rectangularize_court()` — auto court-corner detection (threshold → morphology → contour → convex hull → polygon). **Auto homography setup, no manual clicks** |
| `rectify_court.py:160–209` | `homography(rect, image)` — `cv2.getPerspectiveTransform` + `warpPerspective` → top-down court. Saves `Rectify1.npy`/`RectifyL.npy`/`RectifyR.npy` matrices + `rectified.png` |
| `player_detection.py:15–19` | **HSV team color ranges**: green `[56,50,50]–[100,255,255]`, red `[0,50,50]–[10,255,255]`. *Tune these per game's actual jersey colors.* |
| `player_detection.py:73–188` | `get_players_pos(M, M1, frame, timestamp, map_2d)` — detect players, transform foot point via homography `M` to top-down `map_2d`, classify team by HSV jersey sample, draw dot at **:175** `cv2.circle(map_2d,(pos),10,p.color,7)`. Returns `frame`, `map_2d` (dots), `map_2d_text` (with numbers) |
| `main.py:42–80` | pipeline flow: panorama → court corners → rectify to 20m×12m → init detector → frame loop |
| `tools/plot_tools.py:5–12` | `plt_plot()` matplotlib helper |
| `Rectify*.npy` | precomputed rectification matrices (camera-specific) |

**This is the most directly reusable jersey-color + court-projection code in the org.** Rank #1 for that piece.

---

## C. Reference-only repos (don't build the demo on these, but useful)

- `AI-Basketball-Shot-Detection-Tracker/shot_detector_low_angle.py:1–150` — clean YOLOv8 ball/player detection + cvzone overlay + `cv2.VideoWriter`. Good detection patterns; **no court projection**.
- `Uball_dual_angle_shot_detection/` & `Uball_dual_angle_fusion/` — shot classification only, **no visualization** (the latter is documented in `01_FUSION_LOGIC.md`).
- `basketball_poc/*.ipynb` — exploratory notebooks, conceptual only. Skip.

---

## How the pieces map to the demo

```
            ┌─────────────────────────── game video (FR or wide angle) ──────────────────────────┐
            │                                                                                     │
   ┌────────▼─────────┐     ┌──────────────────────┐     ┌───────────────────────────────────┐
   │ player detection │ ──▶ │ homography (court     │ ──▶ │ jersey-color team classify        │
   │ (YOLOv11)        │     │  corners → top-down)  │     │ (HSV sample on bbox torso)        │
   │ uball_court_     │     │ BasketTracking-1      │     │ BasketTracking-1                  │
   │  mapping          │     │  rectify_court.py     │     │  player_detection.py:15-19,73-188 │
   └──────────────────┘     └──────────────────────┘     └────────────────┬──────────────────┘
                                                                           │
   ┌──────────────────────────────────────────────────────────────────────▼──────────────────┐
   │ video_stitcher.py:create_stitched_frame()  → [ game frame | top-down court w/ team dots ] │
   │ uball_court_mapping  (swap per-tag color → per-team color)                                │
   └───────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                    │
   ┌────────────────────────────────────────────────▼──────────────────────────────────────────┐
   │ overlay shot events from fused detection_results.json  (timestamp_seconds + outcome)       │
   │ source: Uball_dual_angle_fusion  (see 01_FUSION_LOGIC.md §4)                                │
   └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                       output: one synced demo mp4
```

The build plan that wires these together is **[`03_DEMO_BUILD_PLAN.md`](03_DEMO_BUILD_PLAN.md)**.

## Key open questions for the build session

1. **Homography per camera**: production cameras are static. Establish the court→image homography **once per camera** (4+ court-corner correspondences), cache it. `BasketTracking-1/rectify_court.py:96–157` auto-detects corners; if it's unreliable on our footage, fall back to 4 manual clicks (the `uball_court_mapping` calibration UI does exactly this).
2. **Jersey-color ranges are game-specific**: the HSV ranges in `player_detection.py:15–19` are hard-coded. The build must sample the two actual team colors from a frame (or expose them as config) — don't assume green/red.
3. **Which video angle for the court map**: the far/wide angle (FR or a dedicated wide cam) sees the whole court — best for player→court projection. The near angle is too zoomed. Confirm which S3 angle gives full-court coverage.
4. **Foot point, not bbox center**: project the *bottom-center* of each player bbox (where they touch the floor) through the homography, not the centroid — otherwise tall players map deeper into the court than they stand.
