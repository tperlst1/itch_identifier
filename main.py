
"""
Itch Identifier - Main Application

Run:
    python main.py

Environment variables:
    ITCH_DEVICE=cpu
    ITCH_DEVICE=0
    ITCH_CAMERA=0
    ITCH_ENABLE_DISTANCE=0
"""

import os
import logging

from pathlib import Path

from src.itch_identifier import ItchIdentifier
from src.distance_estimator import DistanceEstimator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ItchIdentifier")

ROOT = Path(__file__).resolve().parent


def main():

    model_path = ROOT / "models" / "best.pt"

    calibration_path = (
        ROOT / "models" / "ground_calibration.npz"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. "
            "Train the model or place best.pt in models/."
        )

    device = os.getenv(
        "ITCH_DEVICE",
        "cpu"
    )

    camera = os.getenv(
        "ITCH_CAMERA",
        "0"
    )

    if camera.isdigit():
        camera = int(camera)

    # Distance estimation is explicitly opt-in.
    enable_distance = (
        os.getenv(
            "ITCH_ENABLE_DISTANCE",
            "0"
        ).lower() in ("1", "true", "yes")
    )

    distance_estimator = None

    if enable_distance:

        if calibration_path.is_file():

            try:

                distance_estimator = DistanceEstimator(
                    calibration_path
                )

                logger.info(
                    "Distance calibration loaded."
                )

            except (ValueError, OSError) as error:

                logger.warning(
                    "Distance estimation disabled: %s",
                    error
                )

        else:

            logger.warning(
                "Calibration file not found. "
                "Distance estimation is disabled."
            )

    application = ItchIdentifier(
        model_path=model_path,
        device=device,
        distance_estimator=distance_estimator,
    )

    logger.info(
        "Starting Itch Identifier with device %s",
        device
    )

    application.capture(
        video_feed_path=camera,
        time_slow_down=0.01,
        display_video=True,
    )


if __name__ == "__main__":
    main()
