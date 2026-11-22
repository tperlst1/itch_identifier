
import cv2
import numpy as np


class DistanceEstimator:

    def __init__(self, calibration_file="models/ground_calibration.npz"):

        calibration = np.load(calibration_file)

        self.homography = calibration["homography"]

        # Ground-plane coordinates are in meters.
        # The user/camera ground reference is (0, 0).

    def estimate(self, bbox):

        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = map(float, bbox)

        # Approximate plant ground-contact point.
        plant_x = (x1 + x2) / 2.0
        plant_y = y2

        image_point = np.array(
            [[[plant_x, plant_y]]],
            dtype=np.float32
        )

        # Convert image coordinates to ground coordinates.
        ground_point = cv2.perspectiveTransform(
            image_point,
            self.homography
        )

        world_x, world_y = ground_point[0][0]

        # Distance from the calibrated user/camera origin.
        distance_m = np.sqrt(
            world_x ** 2 + world_y ** 2
        )

        if not np.isfinite(distance_m):
            return None

        return float(distance_m)
