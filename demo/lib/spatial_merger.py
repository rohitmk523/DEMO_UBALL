"""
SpatialCrossCameraMerger — pure-spatial cross-camera association.

The vendored `CrossCameraMerger` (cross_camera_merger.py) weights appearance 0.7
and spatial 0.3 and gates associations at score > 0.5. Our uniform-dot demo has
NO appearance embeddings (same jerseys, no re-id), so the appearance term is
always 0 and the max attainable score is 0.3 < 0.5 → it would never associate
anything and spawn a fresh global player for every detection every frame.

This subclass overrides ONLY the two scoring methods to use pure spatial
proximity in shared court space. The score is mapped into (0.5, 1.0] when two
points are within `max_player_distance_bev`, so the parent's hardcoded `> 0.5`
gates ("closer than max distance") still work unchanged — we touch no other
vendored logic. See 06_DUAL_CAMERA_FUSION.md §5–6.

Units: positions are court coordinates in CM (demo/lib/court.py space), so
`max_player_distance_bev` is in cm (e.g. 150.0 ≈ the 1.5 m same-player radius).
"""
from __future__ import annotations

import math
from typing import Optional

from .cross_camera_merger import CrossCameraMerger, PlayerDetection


def _spatial_score(distance: float, max_distance: float) -> float:
    """Map a court-space distance to a score.

    distance >= max_distance      -> 0.0   (reject)
    distance in [0, max_distance) -> (0.5, 1.0]  (strictly > 0.5 so the
                                      parent's `> 0.5` gate = "within range")
    """
    if max_distance <= 0 or distance >= max_distance:
        return 0.0
    return 0.5 + 0.5 * (1.0 - distance / max_distance)


class SpatialCrossCameraMerger(CrossCameraMerger):
    """CrossCameraMerger variant that associates by court-space proximity only."""

    def _calculate_association_score(self, player1: PlayerDetection,
                                     player2: PlayerDetection) -> float:
        if player1.bev_position is None or player2.bev_position is None:
            return 0.0
        dx = player1.bev_position[0] - player2.bev_position[0]
        dy = player1.bev_position[1] - player2.bev_position[1]
        return _spatial_score(math.hypot(dx, dy), self.max_player_distance_bev)

    def _find_matching_global_player(self, detection: PlayerDetection,
                                     frame_number: int) -> Optional[str]:
        if detection.bev_position is None:
            return None

        best_score = 0.0
        best_global_id: Optional[str] = None

        for global_id, gp in self.global_players.items():
            if frame_number - gp.last_seen_frame > self.max_frames_missing:
                continue
            # don't double-assign the same camera in the same frame
            if detection.camera_id in gp.camera_detections:
                recent = gp.camera_detections[detection.camera_id]
                if recent.frame_number == frame_number:
                    continue
            if not gp.bev_positions:
                continue

            ax, ay = gp.get_average_bev_position()
            dist = math.hypot(detection.bev_position[0] - ax,
                              detection.bev_position[1] - ay)
            score = _spatial_score(dist, self.max_player_distance_bev)
            if score > best_score:
                best_score = score
                best_global_id = global_id

        return best_global_id if best_score > 0.5 else None
