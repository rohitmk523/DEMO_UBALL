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
