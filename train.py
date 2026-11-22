
"""
Itch Identifier - Model Training

Run:
    python train.py

Optional:
    python train.py --epochs 100 --batch 16 --device cpu
"""

import argparse
import logging
import shutil

from pathlib import Path

from src.itch_identifier import ItchIdentifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("Training")

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset" / "data.yaml"

OUTPUT = ROOT / "runs" / "itch_identifier"

MODEL_DIRECTORY = ROOT / "models"

MODEL_OUTPUT = MODEL_DIRECTORY / "best.pt"


def main():

    parser = argparse.ArgumentParser(
        description="Train the Itch Identifier YOLO model."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("Epochs must be positive.")

    if args.batch <= 0:
        parser.error("Batch size must be positive.")

    if not DATASET.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET}"
        )

    logger.info(
        "Dataset configuration: %s",
        DATASET
    )

    logger.info(
        "Training device: %s",
        args.device
    )

    model, metrics = (
        ItchIdentifier.create_detection_model(
            dataset=DATASET,
            epochs=args.epochs,
            image_size=640,
            batch_size=args.batch,
            output_dir=OUTPUT,
            device=args.device,
        )
    )

    best_weights = (
        OUTPUT
        / "species_detection"
        / "weights"
        / "best.pt"
    )

    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Training weights not found: {best_weights}"
        )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        best_weights,
        MODEL_OUTPUT
    )

    logger.info(
        "Best model copied to: %s",
        MODEL_OUTPUT
    )

    print("\nItch Identifier training completed.")

    print("\nValidation metrics:")
    print(metrics)

    print("\nInference model:")
    print(MODEL_OUTPUT)


if __name__ == "__main__":
    main()
