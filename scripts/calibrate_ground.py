
import cv2
import numpy as np

# Image coordinates of visible ground reference points.
# Replace these example coordinates with coordinates
# measured from your actual camera image.

image_points = np.array([
    [180, 420],
    [460, 420],
    [120, 320],
    [520, 320],
    [60, 240],
    [580, 240]
], dtype=np.float32)

# Corresponding measured ground coordinates in meters.
# Origin (0, 0) represents the user's ground position.

ground_points = np.array([
    [-0.50, 0.50],
    [ 0.50, 0.50],
    [-0.50, 1.00],
    [ 0.50, 1.00],
    [-0.50, 1.50],
    [ 0.50, 1.50]
], dtype=np.float32)

# Calculate image-to-ground transformation.
homography, mask = cv2.findHomography(
    image_points,
    ground_points,
    cv2.RANSAC
)

if homography is None:
    raise RuntimeError("Camera calibration failed.")

# Save calibration.
np.savez(
    "models/ground_calibration.npz",
    homography=homography
)

print("Ground-plane calibration saved.")
