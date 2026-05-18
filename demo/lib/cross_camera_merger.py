"""
Cross-Camera Merging System for Basketball Player Tracking

This module implements cross-camera player association and global track management
to unify player IDs across multiple camera views using appearance features and
spatial proximity in Bird's Eye View (BEV) space.

VENDORED VERBATIM from trackingStudio/trackstudio/processors/cross_camera_merger.py
(see 06_DUAL_CAMERA_FUSION.md §6 — verdict: COPY). Do NOT edit logic here; if
upstream changes, re-vendor. Deps: numpy, cv2, scipy (all present).

IMPORTANT REGIME CAVEAT: this class hardcodes a `> 0.5` association threshold with
0.7 appearance / 0.3 spatial weights. With NO appearance embeddings (our uniform-dot
demo: same jerseys, no re-id features) the max attainable score is 0.3 < 0.5, so
zero cross-camera associations occur and every detection spawns a new global player
each frame. For the demo we therefore use `SpatialCrossCameraMerger` in
`demo/lib/spatial_merger.py`, which subclasses this and overrides only the two
scoring methods for a pure-spatial regime. This file stays untouched for provenance.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

@dataclass
class PlayerDetection:
    """Represents a player detection from a single camera."""
    camera_id: int
    track_id: int
    local_player_id: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    confidence: float
    timestamp: float
    frame_number: int
    appearance_embedding: Optional[np.ndarray] = None
    bev_position: Optional[Tuple[float, float]] = None

@dataclass
class GlobalPlayer:
    """Represents a unified global player across all cameras."""
    global_id: str
    camera_detections: Dict[int, PlayerDetection]  # camera_id -> latest detection
    appearance_embeddings: List[np.ndarray]
    bev_positions: List[Tuple[float, float]]
    confidence_scores: List[float]
    last_seen_frame: int
    first_seen_frame: int
    total_detections: int
    
    def get_average_appearance_embedding(self) -> np.ndarray:
        """Get average appearance embedding across all cameras."""
        if not self.appearance_embeddings:
            return np.zeros(512)  # MobileNet embedding size
        return np.mean(self.appearance_embeddings, axis=0)
    
    def get_average_bev_position(self) -> Tuple[float, float]:
        """Get average BEV position across recent detections."""
        if not self.bev_positions:
            return (0.0, 0.0)
        recent_positions = self.bev_positions[-5:]  # Last 5 positions
        avg_x = np.mean([pos[0] for pos in recent_positions])
        avg_y = np.mean([pos[1] for pos in recent_positions])
        return (float(avg_x), float(avg_y))

class CrossCameraMerger:
    """
    Cross-camera player tracking and association system.
    
    This system takes detections from multiple cameras and associates them
    to create unified global player IDs across all camera views.
    """
    
    def __init__(self, max_player_distance_bev: float = 1.5, 
                 appearance_similarity_threshold: float = 0.5,
                 max_frames_missing: int = 60):
        """
        Initialize the cross-camera merger.
        
        Args:
            max_player_distance_bev: Maximum distance in BEV space to associate players (1.5m for overlapping cameras)
            appearance_similarity_threshold: Minimum cosine similarity (0.5 for different camera angles)
            max_frames_missing: Maximum frames a player can be missing (2s at 30fps for overlapping cameras)
        """
        self.max_player_distance_bev = max_player_distance_bev
        self.appearance_similarity_threshold = appearance_similarity_threshold
        self.max_frames_missing = max_frames_missing
        
        # Global player management
        self.global_players: Dict[str, GlobalPlayer] = {}
        self.next_global_id = 1
        
        # Association history
        self.association_history: deque = deque(maxlen=1000)
        
        # Performance metrics
        self.stats = {
            'total_associations': 0,
            'appearance_based_associations': 0,
            'position_based_associations': 0,
            'new_players_created': 0
        }
        
        logger.info("CrossCameraMerger initialized")
    
    def merge_camera_detections(self, camera1_detections: List[Dict], 
                               camera2_detections: List[Dict],
                               frame_number: int, timestamp: float,
                               homography_matrix_1: np.ndarray = None,
                               homography_matrix_2: np.ndarray = None) -> List[GlobalPlayer]:
        """
        Merge detections from both cameras into global player tracks.
        
        Args:
            camera1_detections: Detections from camera 1
            camera2_detections: Detections from camera 2
            frame_number: Current frame number
            timestamp: Current timestamp
            homography_matrix_1: Homography matrix for camera 1 to BEV
            homography_matrix_2: Homography matrix for camera 2 to BEV
            
        Returns:
            List of updated global players
        """
        
        # Convert detections to internal format
        cam1_players = self._convert_to_player_detections(
            camera1_detections, camera_id=1, frame_number=frame_number, 
            timestamp=timestamp, homography_matrix=homography_matrix_1
        )
        
        cam2_players = self._convert_to_player_detections(
            camera2_detections, camera_id=2, frame_number=frame_number, 
            timestamp=timestamp, homography_matrix=homography_matrix_2
        )
        
        # Perform cross-camera association
        self._associate_players_across_cameras(cam1_players, cam2_players, frame_number)
        
        # Update global player tracks
        all_detections = cam1_players + cam2_players
        self._update_global_players(all_detections, frame_number, timestamp)
        
        # Clean up old/lost players
        self._cleanup_old_players(frame_number)
        
        return list(self.global_players.values())
    
    def _convert_to_player_detections(self, detections: List[Dict], camera_id: int,
                                     frame_number: int, timestamp: float,
                                     homography_matrix: np.ndarray = None) -> List[PlayerDetection]:
        """Convert camera tracker detections to internal PlayerDetection format."""
        
        player_detections = []
        
        for det in detections:
            if det.get('class_name') == 'person':
                # Extract appearance embedding if available
                appearance_embedding = None
                if 'appearance_embedding' in det:
                    appearance_embedding = det['appearance_embedding']
                elif hasattr(det, 'appearance_embedding'):
                    appearance_embedding = det.appearance_embedding
                
                # Transform to BEV coordinates if homography is available
                bev_position = None
                if homography_matrix is not None:
                    center = det['center']
                    bev_position = self._transform_to_bev_coordinates(
                        center[0], center[1], homography_matrix
                    )
                
                player_det = PlayerDetection(
                    camera_id=camera_id,
                    track_id=det['track_id'],
                    local_player_id=det['player_id'],
                    bbox=tuple(det['bbox']),
                    center=tuple(det['center']),
                    confidence=det['confidence'],
                    timestamp=timestamp,
                    frame_number=frame_number,
                    appearance_embedding=appearance_embedding,
                    bev_position=bev_position
                )
                
                player_detections.append(player_det)
        
        return player_detections
    
    def _associate_players_across_cameras(self, cam1_players: List[PlayerDetection],
                                        cam2_players: List[PlayerDetection],
                                        frame_number: int):
        """Associate players between cameras using appearance and spatial features."""
        
        if not cam1_players or not cam2_players:
            return
        
        # Create association matrix
        association_matrix = np.zeros((len(cam1_players), len(cam2_players)))
        
        for i, p1 in enumerate(cam1_players):
            for j, p2 in enumerate(cam2_players):
                score = self._calculate_association_score(p1, p2)
                association_matrix[i, j] = score
        
        # Perform Hungarian algorithm for optimal assignment (simplified greedy approach)
        used_cam2 = set()
        associations = []
        
        # Sort by highest scores first
        candidates = []
        for i in range(len(cam1_players)):
            for j in range(len(cam2_players)):
                if association_matrix[i, j] > 0.5:  # Minimum threshold
                    candidates.append((association_matrix[i, j], i, j))
        
        candidates.sort(reverse=True)  # Highest scores first
        
        for score, i, j in candidates:
            if j not in used_cam2:
                associations.append((i, j, score))
                used_cam2.add(j)
                self.stats['total_associations'] += 1
        
        # Store associations in history
        for i, j, score in associations:
            self.association_history.append({
                'frame': frame_number,
                'cam1_player': cam1_players[i].local_player_id,
                'cam2_player': cam2_players[j].local_player_id,
                'score': score
            })
    
    def _calculate_association_score(self, player1: PlayerDetection, 
                                   player2: PlayerDetection) -> float:
        """
        Calculate association score between two player detections.
        
        Combines appearance similarity and spatial proximity in BEV space.
        """
        
        score = 0.0
        
        # Appearance similarity (if embeddings available)
        if (player1.appearance_embedding is not None and 
            player2.appearance_embedding is not None):
            
            # Calculate cosine similarity
            similarity = 1 - cosine(
                player1.appearance_embedding.flatten(),
                player2.appearance_embedding.flatten()
            )
            
            if similarity > self.appearance_similarity_threshold:
                score += similarity * 0.7  # 70% weight for appearance
                self.stats['appearance_based_associations'] += 1
        
        # Spatial proximity in BEV space (if positions available)
        if (player1.bev_position is not None and 
            player2.bev_position is not None):
            
            distance = np.sqrt(
                (player1.bev_position[0] - player2.bev_position[0])**2 +
                (player1.bev_position[1] - player2.bev_position[1])**2
            )
            
            if distance < self.max_player_distance_bev:
                # Inverse distance scoring (closer = higher score)
                proximity_score = 1.0 - (distance / self.max_player_distance_bev)
                score += proximity_score * 0.3  # 30% weight for proximity
                self.stats['position_based_associations'] += 1
        
        return score
    
    def _update_global_players(self, all_detections: List[PlayerDetection], 
                              frame_number: int, timestamp: float):
        """Update global player tracks with new detections."""
        
        # Group detections by associated global players
        detection_to_global = {}
        unassigned_detections = []
        
        for detection in all_detections:
            # Find if this detection belongs to an existing global player
            best_global_id = self._find_matching_global_player(detection, frame_number)
            
            if best_global_id:
                detection_to_global[detection] = best_global_id
            else:
                unassigned_detections.append(detection)
        
        # Update existing global players
        for detection, global_id in detection_to_global.items():
            global_player = self.global_players[global_id]
            
            # Update detection for this camera
            global_player.camera_detections[detection.camera_id] = detection
            
            # Update appearance embeddings
            if detection.appearance_embedding is not None:
                global_player.appearance_embeddings.append(detection.appearance_embedding)
                # Keep only recent embeddings
                if len(global_player.appearance_embeddings) > 20:
                    global_player.appearance_embeddings = global_player.appearance_embeddings[-20:]
            
            # Update BEV positions
            if detection.bev_position is not None:
                global_player.bev_positions.append(detection.bev_position)
                if len(global_player.bev_positions) > 50:
                    global_player.bev_positions = global_player.bev_positions[-50:]
            
            # Update other attributes
            global_player.confidence_scores.append(detection.confidence)
            global_player.last_seen_frame = frame_number
            global_player.total_detections += 1
        
        # Create new global players for unassigned detections
        for detection in unassigned_detections:
            global_id = f"GlobalPlayer_{self.next_global_id}"
            self.next_global_id += 1
            
            global_player = GlobalPlayer(
                global_id=global_id,
                camera_detections={detection.camera_id: detection},
                appearance_embeddings=[detection.appearance_embedding] if detection.appearance_embedding is not None else [],
                bev_positions=[detection.bev_position] if detection.bev_position is not None else [],
                confidence_scores=[detection.confidence],
                last_seen_frame=frame_number,
                first_seen_frame=frame_number,
                total_detections=1
            )
            
            self.global_players[global_id] = global_player
            self.stats['new_players_created'] += 1
    
    def _find_matching_global_player(self, detection: PlayerDetection, 
                                   frame_number: int) -> Optional[str]:
        """Find the best matching global player for a detection."""
        
        best_score = 0.0
        best_global_id = None
        
        for global_id, global_player in self.global_players.items():
            # Skip if player hasn't been seen recently
            if frame_number - global_player.last_seen_frame > self.max_frames_missing:
                continue
            
            # Check if this camera already has a detection for this global player
            if detection.camera_id in global_player.camera_detections:
                recent_detection = global_player.camera_detections[detection.camera_id]
                if recent_detection.frame_number == frame_number:
                    continue  # Already assigned
            
            score = 0.0
            
            # Appearance matching
            if (detection.appearance_embedding is not None and 
                global_player.appearance_embeddings):
                avg_embedding = global_player.get_average_appearance_embedding()
                similarity = 1 - cosine(
                    detection.appearance_embedding.flatten(),
                    avg_embedding.flatten()
                )
                if similarity > self.appearance_similarity_threshold:
                    score += similarity * 0.7
            
            # Spatial proximity
            if (detection.bev_position is not None and 
                global_player.bev_positions):
                avg_position = global_player.get_average_bev_position()
                distance = np.sqrt(
                    (detection.bev_position[0] - avg_position[0])**2 +
                    (detection.bev_position[1] - avg_position[1])**2
                )
                if distance < self.max_player_distance_bev:
                    proximity_score = 1.0 - (distance / self.max_player_distance_bev)
                    score += proximity_score * 0.3
            
            if score > best_score:
                best_score = score
                best_global_id = global_id
        
        return best_global_id if best_score > 0.5 else None
    
    def _cleanup_old_players(self, frame_number: int):
        """Remove global players that haven't been seen for too long."""
        
        to_remove = []
        for global_id, global_player in self.global_players.items():
            if frame_number - global_player.last_seen_frame > self.max_frames_missing:
                to_remove.append(global_id)
        
        for global_id in to_remove:
            del self.global_players[global_id]
            logger.info(f"Removed old global player: {global_id}")
    
    def _transform_to_bev_coordinates(self, x: float, y: float, 
                                    homography_matrix: np.ndarray) -> Optional[Tuple[float, float]]:
        """Transform image coordinates to BEV coordinates."""
        
        try:
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point.reshape(1, 1, 2), homography_matrix)
            return tuple(transformed[0][0])
        except:
            return None
    
    def get_global_player_for_camera_detection(self, camera_id: int, 
                                             local_player_id: str) -> Optional[str]:
        """Get global player ID for a local camera detection."""
        
        for global_id, global_player in self.global_players.items():
            if camera_id in global_player.camera_detections:
                detection = global_player.camera_detections[camera_id]
                if detection.local_player_id == local_player_id:
                    return global_id
        return None
    
    def get_merged_player_tracks(self) -> Dict[str, Dict]:
        """Get comprehensive player tracks for all global players."""
        
        merged_tracks = {}
        
        for global_id, global_player in self.global_players.items():
            track_data = {
                "global_id": global_id,
                "total_detections": global_player.total_detections,
                "first_appearance": global_player.first_seen_frame,
                "last_appearance": global_player.last_seen_frame,
                "camera_coverage": list(global_player.camera_detections.keys()),
                "average_confidence": np.mean(global_player.confidence_scores) if global_player.confidence_scores else 0.0,
                "bev_trajectory": global_player.bev_positions[-100:] if global_player.bev_positions else [],
                "camera_detections": {}
            }
            
            # Add per-camera detection info
            for camera_id, detection in global_player.camera_detections.items():
                track_data["camera_detections"][camera_id] = {
                    "local_player_id": detection.local_player_id,
                    "track_id": detection.track_id,
                    "last_position": detection.center,
                    "last_confidence": detection.confidence,
                    "last_frame": detection.frame_number
                }
            
            merged_tracks[global_id] = track_data
        
        return merged_tracks
    
    def get_statistics(self) -> Dict:
        """Get merger performance statistics."""
        
        current_stats = self.stats.copy()
        current_stats.update({
            'active_global_players': len(self.global_players),
            'association_history_length': len(self.association_history)
        })
        
        return current_stats
    
    def visualize_global_tracks_bev(self, bev_frame: np.ndarray) -> np.ndarray:
        """Visualize global player tracks on BEV frame."""
        
        vis_frame = bev_frame.copy()
        
        # Define colors for global players
        colors = [
            (255, 100, 100),  # Light blue
            (100, 255, 100),  # Light green
            (100, 100, 255),  # Light red
            (255, 255, 100),  # Light cyan
            (255, 100, 255),  # Light magenta
            (100, 255, 255),  # Light yellow
            (200, 150, 100),  # Light brown
            (150, 200, 100),  # Light olive
        ]
        
        for i, (global_id, global_player) in enumerate(self.global_players.items()):
            color = colors[i % len(colors)]
            
            # Draw trajectory
            if len(global_player.bev_positions) > 1:
                points = [(int(pos[0]), int(pos[1])) for pos in global_player.bev_positions[-20:]]
                
                # Draw trail
                for j in range(1, len(points)):
                    alpha = j / len(points)
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(vis_frame, points[j-1], points[j], trail_color, 2)
            
            # Draw current position
            if global_player.bev_positions:
                current_pos = global_player.bev_positions[-1]
                cv2.circle(vis_frame, (int(current_pos[0]), int(current_pos[1])), 10, color, -1)
                
                # Add global ID label
                cv2.putText(vis_frame, global_id, 
                           (int(current_pos[0]) - 30, int(current_pos[1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_frame