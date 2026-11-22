"""
Itch Identifier - Main Application

Run:
    python main.py

Controls:
    P = Pause/resume
    Q = Quit
"""

from src.itch_identifier import ItchIdentifier


def main():

    application = ItchIdentifier(
        model_path="models/best.pt",
        device="0",
        distance_estimator=None,
    )

    video_source = 0

    application.capture(
        video_feed_path=video_source,
        time_slow_down=0.01,
        display_video=True,
    )


if __name__ == "__main__":
    main()
