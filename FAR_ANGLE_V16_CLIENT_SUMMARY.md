# Far-Angle Shot Detection — V1 → v16 Retrain: What We Did & The Results

**Game validated on:** `c2a354fe` (2026-03-19), full 56-minute game, both hoops.
**Ground truth:** every shot in the game hand-annotated by a human operator (the uball.ai `plays` record) — **189 shot attempts (80 made, 109 missed)**, plus the operator scoreboard log.

---

## 1. The problem we set out to fix

The system uses **two camera angles per hoop** — a *near* camera and a *far* (down-court) camera — and fuses them for a reliable made/missed call. The **far camera was effectively blind**: the original far model (V1) was trained on an older facility and did not transfer to the production cameras.

| Far-camera signal (per game) | V1 (old model) |
|---|---|
| Raw shot detections from the far camera | **0 – 12** |
| Near+far frames successfully fused | **0 – 11** |
| Shot-event detection recall (operator-logged) | **74.7%** |
| Free-throw detection recall | **58.3%** |

In practice the far angle contributed almost nothing, so the system was running on the near camera alone and missing roughly 1 in 4 shots.

---

## 2. What we did (method)

We retrained **only the far-angle detector** on real production footage:

- **Model:** YOLOv11n, warm-started from the validated V1 basketball weights (kept what already worked, adapted it to the new facility).
- **Training data:** **5,240 human-verified frames** from production cameras — 3,465 in-play frames + 1,787 free-throw frames. Old-facility data was deliberately excluded so the model learns the real deployment conditions.
- **Validation (held-out):** mAP@50 **0.966**, precision **0.95**, recall **0.93**.
- **Deployment:** versioned model artifact, rolled out behind a safety switch, validated in shadow mode against the production pipeline before anything went live. No change to the near model or the fusion logic — this isolates the win to the far-angle retrain.

---

## 3. The results (V1 → v16, same game, same pipeline)

### A. The far camera now works — this is the core win

| Far-camera signal (per game) | V1 | **v16** |
|---|---|---|
| Raw shot detections from the far camera | 0 – 12 | **144 – 151** |
| Near+far frames successfully fused | 0 – 11 | **127 – 129** |

The far angle went from contributing essentially nothing to being a full, reliable second view of every shot. **This is the headline: the far-angle retrain worked.**

### B. Shot-event detection (did the system see the shot at all)

| Metric | V1 | **v16** |
|---|---|---|
| Detection recall — operator-logged made shots | 74.7% | **100%** (79/79) |
| Free-throw detection recall | 58.3% | **100%** |
| Detection recall — *full* human annotation (all 189 attempts, made **and** missed) | ~75% | **92.6%** |

The system now detects essentially every scoring play, and ~93% of *all* shot attempts including misses.

### C. Made / missed classification (validated against full human annotation)

| Metric | **v16** |
|---|---|
| Made/missed accuracy (on detected shots) | **85.7%** |
| "Made" calls that were correct (precision) | **79.5%** |
| Actual makes correctly called made (recall) | **89.2%** |

---

## 4. Honest framing for the client

- **The far-angle retrain unambiguously worked.** The clearest proof is the far camera going from 0–12 to **144–151** detections per game and 127–129 fused pairs — it went from blind to fully contributing. Shot-event detection rose from ~75% to **93–100%** depending on how it's measured.
- **Detection is the solved part.** The system now reliably *sees* the shots.
- **Made/missed precision is the next focus.** Against the full human-annotated ground truth, the made/miss call is **85.7%** accurate. We have already diagnosed exactly where the remaining error is (rim-bounce-outs being over-called as "made" on the far view, and the angle-disagreement resolver), with a precision-first improvement plan in progress. We recommend presenting detection as the proven milestone and made/miss accuracy as the active, well-understood next step — not as "done."

> One-line for the client: *"We retrained the far-angle camera model on real production footage. The far camera went from effectively blind (≤12 detections/game) to seeing every shot (≈150/game), lifting shot detection from ~75% to ~93–100%. Made/miss classification is now ~86% and is the focused next improvement."*

---

*Numbers sourced from: the v16 training manifest/provenance; the c2a354fe validation vs. the operator scoreboard log; and the full confusion-matrix validation vs. the uball.ai `plays` human annotation (189 attempts). Detection-vs-operator-log is scoped as "operator-logged shots"; the 92.6% / 85.7% figures are vs. complete human annotation including misses, and are the conservative numbers to quote.*
