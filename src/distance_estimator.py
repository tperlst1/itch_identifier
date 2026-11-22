
"""
Itch Identifier - Distance Estimator

Uses a calibrated ground-plane homography.

Important:
A valid calibration must map image coordinates
to ground-plane coordinates expressed in meters.
"""

from pathlib import Path

import cv2
import numpy as np


class DistanceEstimator:

    def __init__(
        self,
        calibration_file="models/ground_calibration.npz"
    ):

        calibration_path = Path(calibration_file)

        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Calibration file not found: {calibration_path}"
            )

        with np.load(calibration_path) as calibration:

            if "homography" not in calibration:
                raise ValueError(
                    "Calibration file must contain 'homography'."
                )

            self.homography = np.asarray(
                calibration["homography"],
                dtype=np.float64,
            )

        if self.homography.shape != (3, 3):
            raise ValueError(
                "Homography must have dimensions 3x3."
            )

        if not np.isfinite(self.homography).all():
            raise ValueError(
                "Homography contains invalid values."
            )

        if abs(np.linalg.det(self.homography)) < 1e-12:
            raise ValueError(
                "Homography matrix is singular."
            )

    def estimate(self, bbox):

        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = map(float, bbox)

        if not np.isfinite([x1, y1, x2, y2]).all():
            return None

        if x2 <= x1 or y2 <= y1:
            return None

        # Approximate ground-contact point.
        plant_x = (x1 + x2) / 2.0
        plant_y = y2

        image_point = np.array(
            [[[plant_x, plant_y]]],
            dtype=np.float32,
        )

        ground_point = cv2.perspectiveTransform(
            image_point,
            self.homography.astype(np.float32),
        )

        world_x, world_y = ground_point[0][0]

        if not np.isfinite([world_x, world_y]).all():
            return None

        distance_m = float(
            np.hypot(world_x, world_y)
        )

        if not np.isfinite(distance_m):
            return None

        return distance_m
