"""
Itch Identifier - Model Training

Run:
    python train.py
"""

from src.itch_identifier import ItchIdentifier


def main():

    model, metrics = (
        ItchIdentifier.create_detection_model(
            dataset="dataset/data.yaml",
            epochs=100,
            image_size=640,
            batch_size=16,
            output_dir="runs/itch_identifier",
        )
    )

    print("\nItch Identifier training completed.")

    print("\nValidation metrics:")
    print(metrics)

    print("\nTrained model location:")
    print(
        "runs/itch_identifier/species_detection/weights/best.pt"
    )


if __name__ == "__main__":
    main()
