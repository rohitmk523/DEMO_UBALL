# 01 — How the Dual-Angle Fusion Logic Works

This is the precise technical map of the shot-detection fusion system, so it can be improved. Source repo: `~/Cellstrat/GitHub_Repositories/Uball_dual_angle_fusion/`.

---

## 1. Architecture / file map

| Component | Path | Role |
|---|---|---|
| **Orchestrator** | `Uball_dual_angle_fusion/dual_angle_fusion.py` (~1326 lines) | The `DualAngleFusion` class — runs both sub-detectors, fuses their outputs |
| **Near sub-detector** | `Uball_dual_angle_fusion/Uball_near_angle_shot_detection/` | Front/side camera. YOLO + overlap-based classification. ~88% acc, best GT coverage (97.4%). Produces `detection_results.json` |
| **Far sub-detector** | `Uball_dual_angle_fusion/Uball_far_angle_shot_detection/` | Down-court camera. YOLO + line-intersection geometry. ~85% acc, specialized for rim bounces / swishes |
| **Fused output** | `results/MM-DD(gameN-ANGLE)_UUID/detection_results.json` | The unified shot list with full decision metadata |
| **Design docs** | `FUSION_USAGE.md`, `FUSION_ANALYSIS.md`, `FUSION_V2_PLAN.md`, `FUSION_V2_FIX_PLAN.md`, `FUSION_V3_PLAN.md`, `FUSION_COMMANDS.md` | Design history + roadmap (summarized in §5) |

Side A = FR+NR (right hoop), Side B = FL+NL (left hoop). One fusion run per side.

---

## 2. The fusion algorithm (dual_angle_fusion.py, by method + line)

### Detection invocation
- `run_near_angle_detection()` / `run_far_angle_detection()` — **lines 120–356**. Spawn the sub-detector via `subprocess.run()` (cwd = the subproject dir), pass video + model + game_id + angle. Read back the subproject's `detection_results.json` (or fall back to a flat `*_session.json`).

### Temporal matching
- `match_detections()` — **lines 358–422**. The core near↔far pairing:
  - **Offset sync** (line 370): `far_ts_synced = far_ts - offset`. `offset` default `0.0` (line 53–58), loadable from an offset JSON.
  - **Temporal window** (line 379): `self.temporal_window`, default **3.0 s** (line 49; CLI override line 1298).
  - Greedy single-best match: for each near shot, pick the closest unmatched far shot within the window. Output `{matches[], unmatched_near[], unmatched_far[]}`.

### Feature analysis (the "why" behind a fused outcome)
- `check_rim_bounce_agreement()` — lines 426–441. Compares `near.is_rim_bounce` vs `far.bounced_back_out`. Agreement → 0.95 conf, disagreement → 0.7.
- `check_entry_angle_consistency()` — lines 443–468. Near's `entry_angle` vs far's `valid_top_crossings`.
- `analyze_swoosh_speed()` — lines 470–493. Near `post_hoop_analysis` vs far `avg_size_ratio`/`valid_bottom_crossings`. Fast swoosh → made (1.2×), slow → missed (1.15×).
- `calculate_fusion_confidence()` — **lines 495–578**. Feature-weighted V3 score (weights in §3) → adaptive confidence multiplier.
- `classify_shot_type()` — lines 580–642 (V3.1). Maps features → {clean_swish, rim_make, rim_bounce_out, near_rim_miss, clean_miss, uncertain}, returns per-type angle reliability boosts.
- `cross_angle_validation()` — lines 644–682 (V3.3). Flags physically-implausible high-confidence disagreements, adjusts.

### Decision
- `resolve_disagreement()` — **lines 684–777**. When near.outcome ≠ far.outcome: classify shot type → weight "made" vs "missed" votes by feature support × angle reliability → apply rim-bounce penalty → pick higher vote → cross-validate.
- `fuse_matched_pair()` — **lines 781–849**. Runs the above for one pair, emits the enriched fused shot dict.
- `process_unmatched()` — **lines 851–895**. Single-angle (near-only / far-only) shots: keep if above threshold, penalize confidence ×0.9. Two modes (see §3).
- `fuse_detections()` — **lines 897–973**. Top-level: load both JSONs → `match_detections` → fuse matched + process unmatched → sort by ts → write `detection_results.json`.

---

## 3. Tunable knobs (this is what you change to improve accuracy)

| Knob | file:line | Default | Effect |
|---|---|---|---|
| `temporal_window` | constructor :49, CLI :1298 | **3.0 s** | near↔far match tolerance. Too wide → wrong pairs; too narrow → missed pairs |
| `offset` | :53–58 / offset file | **0.0 s** | far-vs-near clock skew. Wrong value collapses matched pairs |
| `prioritize_coverage` | :50, CLI :1299 | **False** | True = high-recall (keep ALL unmatched near, far threshold 0.65); False = high-precision (0.75 both) |
| precision threshold | `process_unmatched` :873 | **0.75** | min conf to keep an unmatched shot (precision mode) |
| recall far threshold | :869 | **0.65** | min conf for unmatched far (recall mode); near = 0.0 (keep all) |
| unmatched penalty | :884 | **×0.9** | single-angle shots get 10% confidence haircut |
| weight: outcome_agreement | `calculate_fusion_confidence` :508 | **0.20** | (V3 lowered from 0.30) |
| weight: rim_bounce_agreement | :509 | **0.35** | (V3 raised — most error-reducing signal) |
| weight: entry_angle_consistency | :510 | **0.12** | |
| weight: swoosh_speed | :511 | **0.18** | (V3 raised) |
| weight: overlap_quality (near) | :512 | **0.05** | |
| weight: line_intersection (far) | :513 | **0.10** | (V3 lowered — was over-trusted) |
| conf multiplier (low score <0.5) | :561–563 | `0.6 + score×0.3` (≤0.75×) | |
| conf multiplier (mid 0.5–0.7) | :564–566 | `0.75 + (s-0.5)×1.05` | |
| conf multiplier (high ≥0.7) | :567–569 | `0.935 + (s-0.7)×0.55` (≤1.1×) | |
| final confidence cap | :574 | **0.99** | |
| swoosh fast-made boost | :489 | **1.2×** | |
| swoosh slow-missed boost | :491 | **1.15×** | |

**Highest-leverage knobs for accuracy work (per the V3 analysis):** the feature weights (`:508–513`) and the line_intersection over-trust. See §5.

---

## 4. Output schema (`detection_results.json`)

```json
{
  "session_info": { "start_time", "near_video", "far_video", "offset", "fusion_version" },
  "statistics":   { "total_shots", "made_shots", "missed_shots",
                    "matched_pairs", "unmatched_near_kept", "unmatched_far_kept" },
  "shots": [
    {
      "timestamp_seconds": 25.8,
      "outcome": "made|missed|undetermined",
      "fusion_method": "v2_agreement|v2_feature_resolution|single_near|single_far",
      "fusion_confidence": 0.0-0.99,
      "outcome_agreement": true,
      "time_diff": 0.15,
      "feature_analysis": { "weighted_score", "base_confidence", "feature_scores{...}" },
      "near_detection":  { "outcome","confidence","entry_angle","is_rim_bounce","weighted_overlap_score","method" },
      "far_detection":   { "outcome","confidence","valid_top_crossings","valid_bottom_crossings","bounced_back_out","avg_size_ratio","method" }
    }
  ]
}
```

For the demo, `shots[].timestamp_seconds` + `outcome` + `fusion_confidence` is what you overlay on the synced video.

**Important:** the fused output has **no player/position data** — only shot events. The court-map (player dots) comes from a *separate* tracking pipeline (see `02_COURT_MAPPING.md`). The demo joins them on the shared video timeline.

---

## 5. Known issues + improvement roadmap (from the FUSION_*.md docs)

- **V1**: 91.2% acc; 4/7 errors were false positives ("made" when actually missed) — rim bounces misread as clean makes.
- **V2.0**: 91.0% acc; 6 false positives, all rim bounces → made. Root cause: rim-bounce weight too low + over-optimistic confidence formula.
- **V2.2**: ~91.97% avg. **89.5% of all errors (17/19) are disagreement cases** (near says X, far says Y).
- **V3 plan (in progress)** targets 95%+ by:
  1. **Shot-type-aware angle reliability** — near is better at rim contact, far at clean swishes; weight per shot type.
  2. **Reduce far line_intersection over-trust** — a ball crossing the rim top doesn't mean "made" (rim bounces cross too). Weight lowered 0.10.
  3. **Stronger rim-bounce detection** — combine near+far bounce signals + oscillation analysis.
  4. **Cross-angle validation** — flag high-confidence disagreements with no physical explanation.

**Where to improve, concretely:** the disagreement path (`resolve_disagreement` :684–777) and the feature weights (`:508–513`) are where ~90% of remaining error lives. The V1.5 far-model retrain (already done — see `04_REFERENCES.md`) fixed the *upstream detection* gap; the *fusion decision* logic is the next frontier and is what V3 targets.
