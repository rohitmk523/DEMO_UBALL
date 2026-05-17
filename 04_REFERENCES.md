# 04 — References: Exact Files, Reuse Verdicts, S3 Paths

Copy-paste index of every reusable piece. Verdicts: **COPY** (use ~verbatim) · **ADAPT** (modify) · **REF** (read for patterns) · **SKIP**.

---

## Court mapping / tracking

| File | Verdict | Notes |
|---|---|---|
| `uball_court_mapping/persistent_id_tracking.py` | **ADAPT** | main pipeline orchestration loop — strip UWB, keep YOLO→ByteTrack→project→stitch |
| `uball_court_mapping/app/services/video_stitcher.py` | **COPY** | side-by-side renderer. `create_stitched_frame()`:133, dot render :204–205. Swap color source to per-team |
| `uball_court_mapping/app/services/player_detector.py` | **COPY** | YOLOv11 person detection |
| `uball_court_mapping/app/services/bytetrack_tracker.py` | **COPY** | ByteTrack tracking |
| `uball_court_mapping/app/services/calibration_integration.py` | **ADAPT** | `cv2.findHomography` — use for manual 4-point fallback |
| `uball_court_mapping/app/services/dxf_parser.py` | **REF** | real court geometry (2460×1730 cm) if you want exact dims |
| `uball_court_mapping/app/services/persistent_id_mapper.py` | **SKIP (this demo)** | per-tag color — bypass entirely, use one constant `PLAYER_DOT_COLOR`. ADAPT to per-team only in the later team-color phase |
| `uball_court_mapping/app/services/sam2_segmenter.py` | **REF** | optional mask refine if dots jitter |
| `BasketTracking-1/rectify_court.py:96–157` | **COPY** | auto court-corner detection |
| `BasketTracking-1/rectify_court.py:160–209` | **COPY** | homography → top-down (`getPerspectiveTransform`/`warpPerspective`) |
| `BasketTracking-1/player_detection.py:15–19` | **DEFER** | HSV team color ranges — not used this demo; RE-TUNE per game in the later team-color phase, don't assume green/red |
| `BasketTracking-1/player_detection.py:73–188` | **ADAPT** | `get_players_pos()` — reuse the project + draw-dot path (:175); **skip the jersey-classify branch** this demo (uniform color). Full COPY incl. classify only in the later phase |
| `BasketTracking-1/main.py:42–80` | **REF** | pipeline flow reference |
| `BasketTracking-1/tools/plot_tools.py:5–12` | **REF** | matplotlib helper |
| `BasketTracking-1/Rectify1.npy` / `RectifyL.npy` / `RectifyR.npy` | **REF** | precomputed matrices (camera-specific — recompute for our cameras) |
| `AI-Basketball-Shot-Detection-Tracker/shot_detector_low_angle.py:1–150` | **REF** | clean YOLO+cvzone+VideoWriter patterns |
| `Uball_tracking/` (whole repo) | **SKIP** | tracking-only, no court projection |
| `trackingStudio/` | **REF** | alternative dual-cam BEV; weaker cross-cam re-ID |
| `basketball_poc/*.ipynb` | **SKIP** | exploratory notebooks only |

---

## Fusion / shot detection

| File | Verdict | Notes |
|---|---|---|
| `Uball_dual_angle_fusion/dual_angle_fusion.py` | **REF** | the fusion orchestrator — see `01_FUSION_LOGIC.md` for the full map. Don't modify for the demo; just consume its `detection_results.json` |
| `Uball_dual_angle_fusion/Uball_{near,far}_angle_shot_detection/` | **REF** | sub-detectors; not needed for demo (we consume fused output) |
| `Uball_dual_angle_fusion/FUSION_V3_PLAN.md` | **REF** | improvement roadmap (disagreement-case errors) |
| `Uball_dual_angle_shot_detection/` | **SKIP** | older shot-classifier, superseded by the fusion repo |

---

## Trained models / weights (the validated v16 far model)

| Asset | Location |
|---|---|
| **v16 far weights (validated, 100% recall on c2a354fe)** | `s3://uball-cv-models/yolov11/v2-prod-far/far/best.pt` (sha256 `73eb79c66c6c…`) |
| v16 manifest | `s3://uball-cv-models/yolov11/v2-prod-far/MANIFEST.json` |
| v1 near weights (unchanged, still good) | `s3://uball-cv-models/yolov11/v2-prod-far/near/best.pt` (sha `4dc41e14751b…`) |
| Local v16 deliverable | `Training_frameworks/Uball Far Angle/deliverables/far_v16_best.pt` |
| v16 training plan + provenance | `Training_frameworks/Uball Far Angle/` (numbered docs) |

The v16 far model: YOLOv11n, trained on 5,240 human-verified production frames (3,465 prod + 1,787 FT, old-facility excluded), warm-started from V1 basketball weights. val mAP@50 0.966 / P 0.95 / R 0.93.

---

## S3 — demo source video + fused shot events

| Asset | S3 |
|---|---|
| c2a354fe game source (4 angles) | `s3://uball-videos-production/court-a/2026-03-19/c2a354fe-eb34-4980-af00/2026-03-19_c2a354fe-eb34-4980-af00_{FL,FR,NL,NR}.mp4` |
| c2a354fe v16 fused/near/far session JSONs (Side A/B) | `s3://uball-cv-results/cv-results/court-a/2026-03-19/12a088eb-be66-4514-91b1/side-{A,B}/{detection_results,near_session,far_session}.json` |
| Event-clip training data (for reference) | `s3://uball-cv-models/training-clips/v1-events/` |

Operator ground truth for c2a354fe (for credibility numbers): Firebase `basketball-games/VrcRMco9Rhgss02wAl0B.logs[]` (79 made-shots; v16 detected 79/79 = 100%).

---

## Production pipeline (context only — demo doesn't touch prod)

| Resource | Value |
|---|---|
| cv-fusion Batch JD (v16 + diagnostic image) | `cv-fusion:7` (MODEL_VERSION=v2-prod-far) / `cv-fusion:8` (same + 4× retry) |
| Env var that selects model version | `CV_MODEL_VERSION` (dispatcher reads this; **not** the JD env — known footgun) |
| Shadow gates | `CV_EMIT_TARGET=cv_logs_staging` + `CV_PLAYS_ENABLED=false` (both Jetsons) |
| Jetson SSH | `developer@100.106.30.98` (jetson-nano-001), `developer@100.87.190.71` (jetson-nano-002), key `gopro-automation-linux/id_rsa`, sudo pw `tigerballlily` |

---

## Repo-rank summary (for "where do I start")

1. **`uball_court_mapping/`** — best side-by-side rendering scaffold (proven `*_stitched.mp4` outputs). Start here for the render pipeline.
2. **`BasketTracking-1/`** — for this demo, lift **only the auto-homography + court-projection** code into the scaffold. Its jersey-color team classifier is the reuse target for the *deferred* team-color phase, not now.
3. `Uball_dual_angle_fusion/` — produces the shot events to overlay (consume `detection_results.json`, don't modify).
4. Everything else — reference or skip.

The build plan that sequences these is **[`03_DEMO_BUILD_PLAN.md`](03_DEMO_BUILD_PLAN.md)**.
