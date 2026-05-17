# DEMO_UBALL — Client Demo: Synced Game Video + Live Court Map

This repo holds the **plan + reference map** for a client-facing demo:

> A processed basketball game video plays **side-by-side** with a top-down 2D court diagram. Players appear as **dots colored by jersey/team**, moving in sync with the video. Shot events (made/missed) from the CV pipeline are overlaid.

It also documents **how the dual-angle fusion logic works** and **where every reusable piece of code lives** across the Uball repos, so the demo can be built by stitching proven components rather than from scratch.

The implementation will happen in a fresh Claude Code session driven by the numbered docs.

---

## Read order

1. **[`01_FUSION_LOGIC.md`](01_FUSION_LOGIC.md)** — how the dual-angle near+far fusion works, the exact files, the tunable knobs, and the known accuracy issues + planned improvements (V3 roadmap)
2. **[`02_COURT_MAPPING.md`](02_COURT_MAPPING.md)** — where the court-mapping / player-tracking code is, how homography + jersey-color team assignment work, which repo is most mature
3. **[`03_DEMO_BUILD_PLAN.md`](03_DEMO_BUILD_PLAN.md)** — the step-by-step plan to build the synced video + court-map demo, reusing proven components
4. **[`04_REFERENCES.md`](04_REFERENCES.md)** — exact file:line index of every reusable piece, per repo, with a "copy verbatim / adapt / build-new" verdict

---

## Context: where this fits

The Uball CV pipeline has two halves:

| Half | Status | What it produces |
|---|---|---|
| **Shot detection** (dual-angle fusion) | ✅ V1.5 just validated — far model retrained on production frames, **75% → 100% detection recall** on the c2a354fe game | made/missed shot events with timestamps + confidence |
| **Player tracking + court mapping** | Multiple working prototypes across 3 repos; **no single productionized demo** yet | per-player image-space tracks; (in the best repo) top-down court coordinates |

The demo combines **both**: it plays the game with shot events annotated *and* shows where every player is on a 2D court, colored by team. That's the "wow" visual for the client — proving the system understands the game spatially, not just shot-by-shot.

---

## The headline shot-detection result (context for the demo's credibility)

The far-angle model was retrained on production-camera frames (the work tracked in `Training_frameworks/Uball Far Angle`). Validated on the c2a354fe game vs. operator ground truth:

| Metric | V1 (old far model) | **v16 (new far model)** |
|---|---|---|
| Detection recall | 74.7% | **100%** (79/79 operator-logged shots) |
| Made/missed classification | 86.4% | 87.3% |
| Far-camera raw detections / game | 0–12 | **144–151** |
| Matched near+far fused pairs | 0–11 | **127–129** |
| Free-throw recall | 58.3% | **100%** |

So the shot-event layer the demo overlays is now trustworthy. The remaining build work is the **spatial / court-map layer**.

---

## Demo success criteria

| Criterion | Target |
|---|---|
| Video + court map render **in sync** (same timeline) | frame-accurate within ~1 frame |
| Players shown as dots, **colored by jersey/team** | 2 distinct colors, correct ≥90% of frames |
| Court map is a clean top-down 2D diagram | recognizable as a basketball court |
| Shot events (made/missed) overlaid at correct timestamps | from the fused `detection_results.json` |
| Runs on a processed (not live) game and produces an output mp4 | one self-contained deliverable file |

Out of scope for the first demo: real-time/live processing, multi-court, UWB hardware tags, perfect cross-camera player re-ID.
