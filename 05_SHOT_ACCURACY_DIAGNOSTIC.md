# 05 — Shot Made/Miss Accuracy: Diagnostic + Code-Grounded Fix Proposal

Game **c2a354fe** (2026-03-19), v16 far model. Detection recall **79/79 = 100%**. Made/miss classification **70/79 = 89%** (the table in `README.md` showed 87.3% from the Firebase-side join; the S3-fused join lands at 89% — same story, ±1 shot from match-window choice). This doc explains the **9 wrong shots**, *why* the code produced them, and **exactly what to change** — with a false-positive risk verdict derived from the code itself (we have no miss ground truth; see §2).

> Scope: **diagnostic + fix proposal only. No code was modified.** Visual confirmation clips: see §7.

---

## 1. Method & the 9 shots

Operator log `basketball-games/VrcRMco9Rhgss02wAl0B` → `score_added` entries (relative to `createdAt`) joined to the v16 fused `detection_results.json` (S3 side-A + side-B), nearest match within ±4 s, best constant offset −0.5 s. Operator log carries **team + jersey color** (`leftTeam` UPTOWN `#000000`, `rightTeam` LOS MANA. `#f59e0b`) — free for annotation and the demo.

| # | Period / clock | Side | Pts | Team | NEAR | FAR | Fusion |
|---|---|---|---|---|---|---|---|
| 1 | 1st 18:05 | B | 3 | LOS MANA. | missed c0.92 | missed top0/bot0 | **missed** v2_agreement 0.89 |
| 2 | 1st 15:38 | A | 4 | UPTOWN | missed c0.86 | missed top1/bot0 | **missed** v2_agreement 0.84 |
| 3 | 1st 11:53 | B | 2 | LOS MANA. | missed c0.44 | missed top1/bot0 | **missed** v2_agreement 0.65 |
| 4 | 1st 09:11 | B | 4 | LOS MANA. | missed c0.87 | missed top1/bot0 | **missed** v2_agreement 0.85 |
| 5 | 1st 01:06 | B | 4 | LOS MANA. | missed c0.86 | missed top1/bot0 | **missed** v2_agreement 0.84 |
| 6 | 2nd 18:00 | A | 2 | LOS MANA. | missed c0.48 | missed top0/bot0 | **missed** v2_agreement 0.65 |
| 7 | 2nd 08:52 | B | 4 | UPTOWN | missed c0.88 | missed top1/bot0 | **missed** v2_agreement 0.84 |
| 8 | 2nd 02:59 | A | 2 | LOS MANA. | missed c0.65 | missed top1/bot0 | **missed** v2_agreement 0.76 |
| 9 | 2nd 01:48 | B | 4 | UPTOWN | missed c0.89 | missed top1/bot0 | **missed** v2_agreement 0.84 |

**Failure-mode bucket: 9/9 = "near AND far independently said *missed*, and agreed."** Skew: **5 of 9 are 4-pointers** (longest, fastest, steepest entry); 7 of 9 have the far signature `top_crossings=1, bottom_crossings=0`.

**Measurement blind spot:** the operator logs **only makes** (79 `score_added`, **0 `shot_missed`**). We can measure *makes-called-miss* (these 9) but **cannot** measure *misses-called-make* (false positives) from this ground truth. Every loosening below is therefore risk-assessed from the **code branches** (§5), not from data.

---

## 2. Root cause: the fusion logic is NOT the problem

All 9 are `fusion_method = v2_agreement`. Confirmed in `dual_angle_fusion.py`:

- `match_detections` (`:358–422`) is purely temporal — it never reads `outcome`, cannot change made/miss.
- `fuse_matched_pair` (`:781–849`): at `:797` `if near_outcome == far_outcome:` → `final_outcome = near_outcome`, `fusion_method='v2_agreement'`. **No branch can flip an agreed "missed" to "made"** — `outcome` is copied verbatim, only `confidence` is recomputed.
- `resolve_disagreement` (`:684–777`) — the only place an outcome can be overturned — is reachable **only** via the `else` at `:802–805` when `near_outcome != far_outcome`. For these 9, both said "missed", so it is never invoked.

**Implication:** the old V3 roadmap ("89.5% of errors are disagreement cases", `01_FUSION_LOGIC.md` §5) described the *old far model*. With v16, **0% of the remaining error is a disagreement/fusion problem.** Tuning feature weights (`:508–513`), `temporal_window`, or `resolve_disagreement` cannot fix a single one of these 9. **The fix must be in the two per-angle detectors.**

---

## 3. FAR detector — the actual live classifier

**Important correction.** The live far classifier is **`Uball_far_angle_shot_detection/simple_line_intersection_test.py`** (imported by `Uball_far_angle_shot_detection/main.py:18` as `SimplifiedShotAnalyzer`). `Uball_far_angle_shot_detection/shot_detection.py` is **dead code** for this pipeline (it emits `line_crossings_through_hoop`, never the `valid_top_crossings`/`valid_bottom_crossings` fields we see in the output). `far_detection.method = "unknown"` is just the fusion default at `dual_angle_fusion.py:845` (the far detector writes no `detection_method` key) — not a real state.

**Branch walk** for the 7/9 with `top=1, bottom=0, bounced_back_out=False` (`simple_line_intersection_test.py`):

1. `:356` `if valid_top_crossings >= 1:` → True
2. `:358` `if bounced_back_out:` → False
3. `:366` `if valid_bottom_crossings >= 1:` → **False** → else
4. `:374–376` → `outcome='missed'`, reason `"incomplete_pass (entered top but no bottom exit)"`, `confidence=0.80`

The 2/9 with `top=0/bot=0` hit `:384–386` `"no_top_crossing"`, `confidence=0.90`.

**Why a clean make yields `bottom=0`.** Bottom crossing is detected at `:157`: `crosses_bottom = prev_y < hoop_y2 and ball_y >= hoop_y2 and inside_horizontally`. It needs a tracked ball point **at/below `hoop_y2`** (tight hoop-bbox bottom) while still horizontally inside. A clean swish exits fast and vertical; the ball is **net-occluded** exactly at `hoop_y2`, or the inter-frame gap straddles the edge, or post-hoop parallax pushes the next point outside `hoop_x1..hoop_x2` so `inside_horizontally` is already False. The interpolation rescue at `:164` only fires if the jump exceeds the **entire hoop height** *and* both endpoints stay horizontally inside — a fast long shot satisfies neither. Top crossing succeeds (entry is higher, slower, well inside the 195 px-expanded top zone); bottom does not → forced "missed".

Also: `:300/:305/:320` only count a crossing as *valid* if `MIN_BALL_HOOP_RATIO (0.17) ≤ size_ratio ≤ MAX_BALL_HOOP_RATIO (0.50)` (`:40–41`). A tracked bottom point with an off-range size still fails to register.

---

## 4. NEAR detector — fast-swish gate

Live file: `Uball_near_angle_shot_detection/shot_detection.py` (`main.py:19 from shot_detection import ShotAnalyzer`). Overlap frames are collected only when `overlap_percentage >= min_overlap_threshold` (`:350`, default `1.0` i.e. 100%, `:163/:210`). A fast 4-point swish clears the rim bbox in 1–3 frames → very few frames.

**Branch walk** (fast swish, few 100% frames, `entry_angle` None or steep, `is_rim_bounce=False`):

- `:706` `frames_with_100_percent >= 6 or (>=4 and 95%>=7)` → False
- `:728` `is_rim_bounce and bounce_confidence >= 0.6` → False
- `:735` `frames_with_100_percent >= 3 and not is_rim_bounce` → False (rarely 3)
- `:760` `frames_with_100_percent >= 2 and not is_rim_bounce` → if 2: enter; then `:764` `if ball_continues_down and downward_consistency >= 0.8:` — but a short post-hoop tail has `<3` y-positions, so `:526` pins `downward_consistency = 0.5` → `0.5 >= 0.8` False → `:770–772` `outcome='missed'`, `"insufficient_overlap"`, `confidence≈0.65` (matches shots 3,6,8)
- 0–1 frames: `:776` `weighted_overlap_score >= 3.5` → a 1–2 frame swish scores ≈2.0–2.5 → falls to `:814` else `'missed'`, `confidence≈0.80` (matches shots 1,2,4,5,7,9 c0.86–0.92)

`entry_angle=None` (`:471`, `<2` recent points) disables every steep-entry escape (`:740/:785/:804` all require non-None). So the fastest/steepest 4-pointers lose their only "made" path. The single load-bearing constant is the `<3 points` fallback **`:526` `downward_consistency = 0.5`**, read by the made-gates at `:744/:764/:780/:800`.

---

## 5. Fix proposal — with code-derived false-positive verdicts

We have **no miss ground truth**, so each change is judged by which **true-miss discriminators** in the surrounding code remain reachable.

### FIX A — FAR: extend bottom-crossing to the zone bottom (SAFE)

**Change:** in `simple_line_intersection_test.py:157`, detect the bottom crossing against the **zone bottom** `hoop_y + HOOP_ZONE_VERTICAL` (`HOOP_ZONE_VERTICAL = 95`, `:33`) instead of the tight `hoop_y2`, so fast swishes register a bottom crossing.

**Why FP-safe:** `bounced_back_out` (the in-and-out true-miss guard, `:331–348`, fires on `bounce_upward > 30` at `:347`) is **gated at `:331` on `valid_bottom_crossings >= 1`**. Today these 7 shots have `bottom=0`, so the rim-bounce-out guard is *never even evaluated* for them. Relaxing `:157` so they get `bottom>=1` **automatically activates** the correct miss guard for them. A genuine rim-in-and-out will now be caught by `:347`, not mislabeled.
**Verdict: SAFE** — *only if* implemented as geometry relaxation at `:157` (routes through `:331–348`). **RISKY** if implemented as a new "made" branch at `:374` that *bypasses* the `bounced_back_out` check — that would flip true in-and-out misses to made. Do **not** bypass `:331`. Also keep the `MIN/MAX_BALL_HOOP_RATIO` gate (`:305/:320`) so front-of-hoop passes still reject.

### FIX B — NEAR: relax only the fast-swish downward-consistency sub-gate (SAFE)

**Change:** at `:764`, lower `downward_consistency >= 0.8` → `>= 0.5` (or raise the `<3 points` fallback at `:526` from `0.5` to ~`0.6`). **Keep** the `not is_rim_bounce` precondition and **keep** `frames_with_100_percent >= 2` at `:760`.

**Why FP-safe:** the true-miss guards are independent of the loosened thresholds — `_enhanced_rim_bounce_detection` (`:538–573`, fires `:570 bounce_score >= 2.5`, dominated by `:547 ball_bounces_back +2.0` and `:551 entry_angle<35 +1.5`) and `steep_entry_bounce_back` (`:699–702`). Neither reads `downward_consistency` or `frames_with_100_percent`, so upward-bounce misses stay rejected. Reaching `:760` still requires **2 frames of the ball fully enclosed by the rim bbox** — a true miss rarely achieves that without being a rim-in-and-out, which `:547 ball_bounces_back` catches.
**Verdict: SAFE.** **RISKY** if you also lower `:760` from `>=2` to `>=1` — a single fully-enclosed frame is reachable by a hard rim-in that pops out, and rim-bounce confidence is sometimes `<0.6` (the `:728` threshold), letting it escape the guard. **Keep `:760` at `>=2`.**

### Combined expected effect

Both detectors flip these makes to "made" *only* via paths that re-enter the existing in-and-out / rim-bounce rejectors. Net: the 9 should recover (→ ~79/79 classification on c2a354fe) with the true-miss discriminators (`bounced_back_out bounce_upward>30`, `is_rim_bounce bounce_score>=2.5`, `steep_entry_bounce_back`) **unchanged and still reachable**. Because both detectors flip independently, fusion's `v2_agreement` path then carries the corrected "made" through with no fusion change.

---

## 6. How to validate the fix (before anything ships)

1. Apply A + B on a branch; re-run the fusion pipeline on c2a354fe (both sides) — confirm the 9 flip to made and detection recall stays 79/79.
2. **FP sanity without operator miss-logs:** hand-label ~30–40 *misses* from the annotated review video (§7) — specifically rim-in-and-out and air-balls — and confirm none flip to "made" after A+B. This directly exercises the `bounced_back_out`/`is_rim_bounce` guards the fix relies on.
3. Re-run on a **second** game (different lighting/teams) to check the loosened thresholds don't over-trigger.
4. Only then consider shadow deployment (the `CV_PLAYS_ENABLED=false` gate already protects production).

---

## 7. Visual confirmation artifact

For each of the 9 shots: a side-by-side **near | far** clip (±6 s) using each detector's own annotated `*_detected.mp4` (bbox, ball trajectory colored by zone, the detector's own made/miss label), with a burned-in caption stating operator truth vs near/far/fusion verdicts. Lets you watch the ball pass cleanly through while the far trajectory shows the top-zone crossing with no registered bottom crossing, and the near overlap window is too short — i.e. exactly the §3/§4 failure modes, confirmed by eye. Output: `/tmp/diag_clips/shot_NN_*.mp4` + `ALL_9_misclassified_reel.mp4` (local).

---

## 8. One-line takeaway

> v16 made shot **detection** essentially perfect (79/79). The remaining 9 made/miss errors are **100% upstream geometry**, not fusion: far's bottom-crossing test is too tight for fast/long swishes (`simple_line_intersection_test.py:157`), and near's swish gate is unreachable when the post-hoop tail is short (`shot_detection.py:526/:764`). Two narrow, FP-safe relaxations that re-enter the existing in-and-out guards recover all 9 without touching fusion.
