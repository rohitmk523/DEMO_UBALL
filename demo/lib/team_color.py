"""
Jersey team classification (auto, 2 teams + referee/neutral).

This facility plays on a TAN wood floor with a RED painted key, and the
two squads here are a DARK (black/maroon) kit vs a WHITE/LIGHT kit -- the
discriminating signal is jersey BRIGHTNESS, not hue (red floor + skin +
maroon jerseys all collide in hue).  So the descriptor is the torso's
robust (value, saturation) after masking out the wood-floor tan / skin
and the bright court reflections, and teams are separated by clip-level
k-means(k=2) in that 2-D space.

Strategy (clip-level, NOT per-frame so labels are temporally stable):
  1. Crop the chest ROI (upper torso, kept high to dodge the floor),
     mask wood-tan / skin hue and floor glare, take robust median V & S
     of the remaining jersey pixels + a darkness fraction.
  2. Accumulate per fused-track id over the whole clip.
  3. Periodically k-means(2) the per-track median (V,S) -> dark vs light
     team centroids.  A track far from BOTH (or ambiguous) -> ref.
  4. Per-track label = nearest centroid, LOCKED with hysteresis once
     `lock_samples` confident samples seen so it never flips per frame.

Public:
  TeamClassifier  - stateful, fed crops, queried for a track's label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LABEL_A, LABEL_B, LABEL_REF = "A", "B", "REF"


# --------------------------------------------------------------------------- #
#  Torso descriptor  ->  (value, saturation, hue, jersey_fraction)
# --------------------------------------------------------------------------- #
def torso_descriptor(
    frame: np.ndarray,
    box: Tuple[float, float, float, float],
    skin_hue_lo: int,
    skin_hue_hi: int,
    min_sat: int,
    min_val: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (V, S, H, jersey_fraction) for the chest ROI, or None.

    jersey_fraction = kept px / roi px (confidence proxy: low when the
    crop is mostly floor / skin / glare).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 8:
        return None
    # Chest band: kept HIGH (0.12..0.42 h) so the tan floor seen between
    # the legs / under a leaning torso does not pollute the colour.
    rx1 = int(round(x1 + 0.22 * bw))
    rx2 = int(round(x2 - 0.22 * bw))
    ry1 = int(round(y1 + 0.12 * bh))
    ry2 = int(round(y1 + 0.42 * bh))
    rx1, rx2 = max(0, rx1), min(w, rx2)
    ry1, ry2 = max(0, ry1), min(h, ry2)
    if rx2 - rx1 < 3 or ry2 - ry1 < 3:
        return None
    roi = frame[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)
    total = H.size
    if total == 0:
        return None
    # Drop wood-tan / skin hue ONLY when it is also low-value (the actual
    # floor / bare skin); a bright saturated tan pixel could be a jersey.
    tan = (H >= skin_hue_lo) & (H <= skin_hue_hi) & (S < 150)
    glare = V > 245                           # specular court reflection
    keep = (~tan) & (~glare)
    nkeep = int(keep.sum())
    if nkeep < max(15, int(0.08 * total)):
        return None
    vk = V[keep]
    sk = S[keep]
    hk = H[keep].astype(np.float32)
    vmed = float(np.median(vk))
    smed = float(np.median(sk))
    ang = np.deg2rad(hk * 2.0)
    cm = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    hmed = (np.rad2deg(cm) % 360.0) / 2.0
    return vmed, smed, hmed, nkeep / float(total)


# --------------------------------------------------------------------------- #
#  Per-track accumulation + clip-level clustering (dark vs light)
# --------------------------------------------------------------------------- #
@dataclass
class _Acc:
    vals: List[float] = field(default_factory=list)
    sats: List[float] = field(default_factory=list)
    hues: List[float] = field(default_factory=list)
    locked: Optional[str] = None

    def add(self, v: float, s: float, h: float) -> None:
        self.vals.append(v)
        self.sats.append(s)
        self.hues.append(h)

    def median(self) -> Optional[Tuple[float, float, float]]:
        if not self.vals:
            return None
        ang = np.deg2rad(np.asarray(self.hues) * 2.0)
        cm = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
        hm = (np.rad2deg(cm) % 360.0) / 2.0
        return (float(np.median(self.vals)),
                float(np.median(self.sats)), hm)


class TeamClassifier:
    """Clip-level dark-vs-light team classifier, stable hysteretic labels.

    Constructor keeps the historical hue-style kwarg names so the CLI
    surface is unchanged; here they gate the tan/skin mask and the
    ambiguous-track -> ref decision.
    """

    def __init__(
        self,
        skin_hue_lo: int = 3,
        skin_hue_hi: int = 24,
        min_sat: int = 55,
        min_val: int = 45,
        ref_min_sat: int = 50,
        ref_hue_margin: float = 22.0,
        lock_samples: int = 10,
        recluster_every: int = 25,
    ):
        self.skin_hue_lo = skin_hue_lo
        self.skin_hue_hi = skin_hue_hi
        self.min_sat = min_sat
        self.min_val = min_val
        # ref_hue_margin reused as the (V,S) ambiguity band half-width.
        self.amb_band = max(8.0, ref_hue_margin)
        self.lock_samples = lock_samples
        self.recluster_every = recluster_every
        self.acc: Dict[int, _Acc] = {}
        # centroids in (V, S) space; A = darker, B = lighter.
        self.cen: Optional[Tuple[Tuple[float, float],
                                 Tuple[float, float]]] = None
        self._since = 0

    # ---- ingestion -------------------------------------------------- #
    def observe(self, tid: int, frame: np.ndarray,
                box: Tuple[float, float, float, float]) -> None:
        d = torso_descriptor(frame, box, self.skin_hue_lo,
                             self.skin_hue_hi, self.min_sat, self.min_val)
        if d is None:
            return
        v, s, h, _fr = d
        self.acc.setdefault(tid, _Acc()).add(v, s, h)
        self._since += 1
        if self._since >= self.recluster_every:
            self._recluster()
            self._since = 0

    # ---- clustering (1-D k-means on value) -------------------------- #
    def _recluster(self) -> None:
        vs: List[float] = []
        ss: List[float] = []
        for a in self.acc.values():
            if len(a.vals) < 4:
                continue
            m = a.median()
            if m is None:
                continue
            vs.append(m[0])
            ss.append(m[1])
        if len(vs) < 2:
            return
        arr = np.asarray(vs, np.float32)
        sarr = np.asarray(ss, np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 12.0:
            return                                # not separable yet
        c0, c1 = lo, hi                           # dark seed / light seed
        for _ in range(30):
            g0 = arr <= (c0 + c1) / 2.0
            g1 = ~g0
            if not g0.any() or not g1.any():
                break
            n0 = float(arr[g0].mean())
            n1 = float(arr[g1].mean())
            if abs(n0 - c0) < 0.3 and abs(n1 - c1) < 0.3:
                c0, c1 = n0, n1
                break
            c0, c1 = n0, n1
        g0 = arr <= (c0 + c1) / 2.0
        s0 = float(np.median(sarr[g0])) if g0.any() else 120.0
        s1 = float(np.median(sarr[~g0])) if (~g0).any() else 120.0
        # A = darker team, B = lighter team (stable mapping).
        self.cen = ((c0, s0), (c1, s1))

    # ---- query ------------------------------------------------------ #
    def _classify_median(
        self, m: Tuple[float, float, float]
    ) -> str:
        if self.cen is None:
            return LABEL_REF
        v, _s, _h = m
        (va, _sa), (vb, _sb) = self.cen
        mid = (va + vb) / 2.0
        # Ambiguous (sits on the dark/light boundary) -> neutral / ref.
        if abs(v - mid) < self.amb_band * 0.30 and (vb - va) > 0:
            return LABEL_REF
        return LABEL_A if abs(v - va) <= abs(v - vb) else LABEL_B

    def label(self, tid: int) -> str:
        a = self.acc.get(tid)
        if a is None:
            return LABEL_REF
        if a.locked is not None:
            return a.locked
        m = a.median()
        if m is None:
            return LABEL_REF
        lab = self._classify_median(m)
        if len(a.vals) >= self.lock_samples and self.cen is not None:
            a.locked = lab
        return lab

    def finalize(self) -> None:
        self._recluster()

    # ---- display ---------------------------------------------------- #
    def team_bgr(self, label: str) -> Tuple[int, int, int]:
        """Dot/trail fill colour — matches the real jersey."""
        if label == LABEL_REF:
            return (180, 180, 180)             # light gray (refs/neutral)
        if label == LABEL_A:                  # dark team -> BLACK jersey
            return (0, 0, 0)
        return (255, 255, 255)                # light team -> WHITE jersey

    def team_outline_bgr(self, label: str) -> Tuple[int, int, int]:
        """Outline ring colour for visibility on the dark court canvas."""
        if label == LABEL_A:                  # black dot -> white ring
            return (255, 255, 255)
        if label == LABEL_B:                  # white dot -> black ring
            return (0, 0, 0)
        return (40, 40, 40)                   # ref grey -> dark ring


# ===================================================================== #
#  SigLIP-embedding team classifier (drop-in replacement)               #
# ===================================================================== #

class SigLIPTeamClassifier:
    """Team classifier using SigLIP image embeddings + k-means.

    Matches the public API of `TeamClassifier` (observe / label / finalize
    / team_bgr / team_outline_bgr) so make_fusion_demo.py can swap by
    --team-mode. Uses google/siglip-base-patch16-224 from HuggingFace.

    Inspired by https://blog.roboflow.com/identify-basketball-players/
    (their team-classification step). Robust to the dark-vs-white-on-
    red-floor case where HSV/brightness clustering fails: semantic
    embeddings cluster jerseys cleanly regardless of court colour.
    """

    MODEL_ID = "google/siglip-base-patch16-224"

    def __init__(
        self,
        sample_every: int = 4,
        lock_samples: int = 6,
        recluster_every: int = 25,
        min_tracks_for_cluster: int = 3,
        device: str = "mps",
        # legacy kwargs accepted+ignored so the CLI surface stays uniform
        **_legacy,
    ) -> None:
        self.sample_every = max(1, sample_every)
        self.lock_samples = max(2, lock_samples)
        self.recluster_every = max(5, recluster_every)
        self.min_tracks = max(2, min_tracks_for_cluster)
        self.device = device
        self.calls: Dict[int, int] = {}
        self.embeds: Dict[int, List[np.ndarray]] = {}
        self.locked: Dict[int, str] = {}              # tid -> LABEL_*
        self.centroids: Optional[np.ndarray] = None    # (2, D)
        self._since = 0
        self._mdl = None
        self._proc = None
        self._torch = None

    # ---- lazy model load ------------------------------------------- #
    def _lazy(self) -> None:
        if self._mdl is not None:
            return
        from transformers import AutoProcessor, AutoModel  # type: ignore
        import torch                                       # type: ignore
        self._proc = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._mdl = AutoModel.from_pretrained(
            self.MODEL_ID).to(self.device).eval()
        self._torch = torch

    # ---- ingestion ------------------------------------------------- #
    def observe(self, tid: int, frame: np.ndarray,
                box: Tuple[float, float, float, float]) -> None:
        if self.locked.get(tid) is not None:           # already final
            return
        self.calls[tid] = self.calls.get(tid, 0) + 1
        if self.calls[tid] % self.sample_every != 1:
            return
        roi = _torso_roi_for_siglip(frame, box)
        if roi is None or roi.size == 0:
            return
        self._lazy()
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        inp = self._proc(images=rgb, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with self._torch.no_grad():                    # type: ignore
            out = self._mdl.get_image_features(**inp)   # type: ignore
        # transformers 5.x may return a ModelOutput rather than a tensor
        if hasattr(out, "cpu"):
            vec = out
        else:
            vec = getattr(out, "image_embeds", None)
            if vec is None:
                vec = getattr(out, "pooler_output", None)
            if vec is None:
                vec = out.last_hidden_state.mean(dim=1)
        emb = vec.cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        self.embeds.setdefault(tid, []).append(emb)
        if len(self.embeds[tid]) > 24:
            self.embeds[tid] = self.embeds[tid][-24:]
        self._since += 1
        if self._since >= self.recluster_every:
            self._recluster()
            self._since = 0
            self._lock_ready()

    # ---- clustering ------------------------------------------------ #
    def _recluster(self) -> None:
        ready = [(t, np.mean(es, axis=0))
                 for t, es in self.embeds.items() if len(es) >= 2]
        if len(ready) < self.min_tracks:
            return
        from sklearn.cluster import KMeans              # type: ignore
        X = np.stack([m for _, m in ready])
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        # Stabilise A/B mapping: A = first centroid by raw-image-mean
        # brightness of its members (darker -> A) so colours match the
        # brightness-classifier convention.
        bri = np.zeros(2, np.float32)
        cnt = np.zeros(2, np.float32)
        for (t, _), lbl in zip(ready, km.labels_):
            # use the L2 norm of the descriptor itself as a cheap proxy
            # — not perfect, but consistent. (Sign-flip below ensures the
            # final A/B label assignment isn't dependent on KMeans init.)
            bri[lbl] += float(np.mean(self.embeds[t][-1]))
            cnt[lbl] += 1.0
        cnt = np.maximum(cnt, 1)
        avg = bri / cnt
        cen = km.cluster_centers_.copy()
        if avg[0] > avg[1]:                  # ensure cluster 0 = darker proxy
            cen = cen[::-1]
        self.centroids = cen

    def _lock_ready(self) -> None:
        if self.centroids is None:
            return
        for tid, es in self.embeds.items():
            if tid in self.locked or len(es) < self.lock_samples:
                continue
            m = np.mean(es, axis=0)
            d0 = float(np.linalg.norm(m - self.centroids[0]))
            d1 = float(np.linalg.norm(m - self.centroids[1]))
            self.locked[tid] = LABEL_A if d0 < d1 else LABEL_B

    # ---- referee override ----------------------------------------- #
    def mark_ref(self, tid: int) -> None:
        """The detector said this track is a referee — lock directly."""
        self.locked[tid] = LABEL_REF

    # ---- query ---------------------------------------------------- #
    def label(self, tid: int) -> str:
        if tid in self.locked:
            return self.locked[tid]
        # provisional label while warming up
        es = self.embeds.get(tid)
        if es is None or self.centroids is None:
            return LABEL_REF
        m = np.mean(es, axis=0)
        d0 = float(np.linalg.norm(m - self.centroids[0]))
        d1 = float(np.linalg.norm(m - self.centroids[1]))
        return LABEL_A if d0 < d1 else LABEL_B

    def finalize(self) -> None:
        self._recluster()
        self._lock_ready()

    # ---- display (shared with TeamClassifier) --------------------- #
    team_bgr = TeamClassifier.team_bgr
    team_outline_bgr = TeamClassifier.team_outline_bgr


def _torso_roi_for_siglip(
    frame: np.ndarray, box: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """Central upper-body crop (jersey) for SigLIP embedding.

    Slightly looser than `torso_descriptor`'s ROI — SigLIP benefits from
    a bit of jersey context (sleeves, shoulders).
    """
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    if h < 48 or w < 24:                       # too tiny -> skip
        return None
    cx1 = int(x1 + 0.15 * w)
    cx2 = int(x2 - 0.15 * w)
    cy1 = int(y1 + 0.08 * h)
    cy2 = int(y1 + 0.55 * h)
    H, W = frame.shape[:2]
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(W, cx2), min(H, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2]
