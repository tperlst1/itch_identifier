
"""
Itch Identifier - Main Application

Controls:
    P = Pause/resume
    Q = Quit

Run:
    python main.py
"""

import logging
import os

from pathlib import Path

from src.itch_identifier import ItchIdentifier
from src.distance_estimator import DistanceEstimator


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ItchIdentifier")


def main():

    model_path = Path("models/best.pt")

    calibration_path = Path(
        "models/ground_calibration.npz"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model missing: {model_path}"
        )

    distance_estimator = None

    if calibration_path.is_file():

        try:

            distance_estimator = DistanceEstimator(
                calibration_path
            )

            logger.info(
                "Distance calibration loaded."
            )

        except (ValueError, OSError, KeyError) as error:

            logger.warning(
                "Distance estimation disabled: %s",
                error
            )

    else:

        logger.warning(
            "No calibration found. "
            "Distance-based proximity alerts are disabled."
        )

    device = os.getenv(
        "ITCH_DEVICE",
        "cpu"
    )

    application = ItchIdentifier(
        model_path=model_path,
        device=device,
        distance_estimator=distance_estimator,
    )

    video_source = int(
        os.getenv("ITCH_CAMERA", "0")
    )

    application.capture(
        video_feed_path=video_source,
        time_slow_down=0.01,
        display_video=True,
    )


if __name__ == "__main__":
    main()
