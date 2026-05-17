"""
Calibration integration for projecting court coordinates to image coordinates.

VENDORED VERBATIM from uball_court_mapping/app/services/calibration_integration.py
(see DEMO_UBALL/04_REFERENCES.md — verdict: COPY). Do not edit logic here; if the
upstream changes, re-vendor. cv2 4.12 compatible (findHomography / perspectiveTransform,
no xfeatures2d). This is the homography core for 03_DEMO_BUILD_PLAN.md Step 1.

Court coordinate system used by this class: NBA full court 2865 cm (length, X) x
1524 cm (width, Y), origin at one baseline corner. The demo treats this as the
canonical top-down court space; metric exactness is not required for the
uniform-dot demo, only internal consistency.
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class CalibrationIntegration:
    """Handles court-to-image coordinate projection using calibration matrix."""

    # Basketball court dimensions (in cm)
    # UWB system uses: X along long side (2865 cm), Y along short side (1524 cm)
    COURT_LENGTH_CM = 2865.0  # 94 feet = 28.65 meters
    COURT_WIDTH_CM = 1524.0   # 50 feet = 15.24 meters

    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize calibration integration.

        Args:
            calibration_file: Path to calibration results JSON file
        """
        self.homography_matrix = None
        self.inverse_homography = None
        self.court_points = None
        self.image_points = None

        if calibration_file:
            self.load_calibration(calibration_file)

    def load_calibration(self, calibration_file: str) -> bool:
        """
        Load calibration matrix from file.

        Args:
            calibration_file: Path to calibration JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            calibration_path = Path(calibration_file)
            if not calibration_path.exists():
                logger.error(f"Calibration file not found: {calibration_file}")
                return False

            with open(calibration_path, 'r') as f:
                data = json.load(f)

            # Extract homography matrix (support both formats)
            if 'homography_matrix' in data:
                self.homography_matrix = np.array(data['homography_matrix'])
            elif 'homography' in data:
                self.homography_matrix = np.array(data['homography'])

            if self.homography_matrix is not None:
                self.inverse_homography = np.linalg.inv(self.homography_matrix)
                logger.info("Loaded homography matrix from calibration file")
            else:
                logger.warning("No homography matrix found in calibration file")

            # Extract calibration points if available (support both formats)
            if 'court_points' in data:
                self.court_points = np.array(data['court_points'])
                # Support both 'image_points' and 'video_points'
                if 'image_points' in data:
                    self.image_points = np.array(data['image_points'])
                elif 'video_points' in data:
                    self.image_points = np.array(data['video_points'])

                if self.image_points is not None:
                    logger.info(f"Loaded {len(self.court_points)} calibration points")

            return True

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    def rotate_uwb_to_calibration(
        self,
        x_uwb: float,
        y_uwb: float
    ) -> Tuple[float, float]:
        """
        Rotate UWB coordinates 90° counter-clockwise to match calibration orientation.

        UWB System: X along long side (left-right), Y along short side (bottom-top)
        Calibration: X along short side (left-right), Y along long side (bottom-top)

        Rotation: 90° counter-clockwise
        - x_cal = COURT_LENGTH - x_uwb
        - y_cal = y_uwb

        Args:
            x_uwb: X coordinate in UWB system (cm) - along long side
            y_uwb: Y coordinate in UWB system (cm) - along short side

        Returns:
            Tuple of (x_cal, y_cal) in calibration coordinate system
        """
        x_cal = self.COURT_LENGTH_CM - x_uwb
        y_cal = y_uwb

        return x_cal, y_cal

    def compute_homography(
        self,
        court_points: List[Tuple[float, float]],
        image_points: List[Tuple[float, float]]
    ) -> bool:
        """
        Compute homography matrix from point correspondences.

        Args:
            court_points: List of (x, y) court coordinates in cm
            image_points: List of (x, y) image coordinates in pixels

        Returns:
            True if successful, False otherwise
        """
        try:
            if len(court_points) != len(image_points) or len(court_points) < 4:
                logger.error("Need at least 4 point correspondences")
                return False

            self.court_points = np.array(court_points, dtype=np.float32)
            self.image_points = np.array(image_points, dtype=np.float32)

            # Compute homography using OpenCV
            self.homography_matrix, mask = cv2.findHomography(
                self.court_points,
                self.image_points,
                cv2.RANSAC,
                5.0
            )

            if self.homography_matrix is None:
                logger.error("Failed to compute homography matrix")
                return False

            self.inverse_homography = np.linalg.inv(self.homography_matrix)
            logger.info("Computed homography matrix")
            return True

        except Exception as e:
            logger.error(f"Failed to compute homography: {e}")
            return False

    def uwb_to_image(self, x_uwb: float, y_uwb: float) -> Tuple[int, int]:
        """
        Project UWB coordinates to image coordinates (with rotation).

        This method handles the coordinate system mismatch:
        - UWB uses X along long side, Y along short side
        - Calibration uses X along short side, Y along long side

        Args:
            x_uwb: X coordinate in UWB system (cm) - along long side
            y_uwb: Y coordinate in UWB system (cm) - along short side

        Returns:
            Tuple of (pixel_x, pixel_y) in image space
        """
        # Step 1: Rotate coordinates to match calibration orientation
        x_cal, y_cal = self.rotate_uwb_to_calibration(x_uwb, y_uwb)

        # Step 2: Project to image using homography
        return self.court_to_image(x_cal, y_cal)

    def court_to_image(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        """
        Project court coordinates to image coordinates.

        Note: If you have UWB coordinates, use uwb_to_image() instead,
        as it handles the coordinate system rotation.

        Args:
            x_cm: X coordinate in court space (cm)
            y_cm: Y coordinate in court space (cm)

        Returns:
            Tuple of (pixel_x, pixel_y) in image space
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not initialized")

        # Create point in homogeneous coordinates
        point = np.array([[x_cm, y_cm]], dtype=np.float32)

        # Apply homography transformation
        transformed = cv2.perspectiveTransform(
            point.reshape(1, 1, 2),
            self.homography_matrix
        )

        pixel_x = int(transformed[0, 0, 0])
        pixel_y = int(transformed[0, 0, 1])

        return pixel_x, pixel_y

    def court_to_image_batch(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Project multiple court coordinates to image coordinates.

        Args:
            points: List of (x_cm, y_cm) court coordinates

        Returns:
            List of (pixel_x, pixel_y) image coordinates
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not initialized")

        if not points:
            return []

        # Convert to numpy array
        court_pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # Apply homography transformation
        image_pts = cv2.perspectiveTransform(court_pts, self.homography_matrix)

        # Convert to list of tuples
        result = [(int(pt[0, 0]), int(pt[0, 1])) for pt in image_pts]

        return result

    def image_to_court(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Project image coordinates back to court coordinates.

        Args:
            pixel_x: X coordinate in image (pixels)
            pixel_y: Y coordinate in image (pixels)

        Returns:
            Tuple of (x_cm, y_cm) in court space
        """
        if self.inverse_homography is None:
            raise ValueError("Inverse homography not initialized")

        # Create point in homogeneous coordinates
        point = np.array([[float(pixel_x), float(pixel_y)]], dtype=np.float32)

        # Apply inverse homography
        transformed = cv2.perspectiveTransform(
            point.reshape(1, 1, 2),
            self.inverse_homography
        )

        x_cm = transformed[0, 0, 0]
        y_cm = transformed[0, 0, 1]

        return x_cm, y_cm

    def is_point_in_image(
        self,
        pixel_x: int,
        pixel_y: int,
        image_width: int,
        image_height: int
    ) -> bool:
        """
        Check if projected point is within image bounds.

        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            image_width: Image width
            image_height: Image height

        Returns:
            True if point is within bounds
        """
        return (0 <= pixel_x < image_width and 0 <= pixel_y < image_height)

    def save_calibration(self, output_file: str) -> bool:
        """
        Save calibration data to JSON file.

        Args:
            output_file: Path to output JSON file

        Returns:
            True if successful
        """
        try:
            data = {
                "homography_matrix": self.homography_matrix.tolist(),
                "court_points": self.court_points.tolist() if self.court_points is not None else None,
                "image_points": self.image_points.tolist() if self.image_points is not None else None
            }

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved calibration to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
