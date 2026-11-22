"""
Itch Identifier - Backend

Responsibilities:
- YOLO plant detection
- Species and hazard classification
- Optional distance estimation
- Dangerous plant counting
- Proximity and area safety alerts
- Video capture and lifecycle management

The frontend owns visual annotation.
"""

import cv2
import time
import logging
import math

from pathlib import Path
from ultralytics import YOLO

from src import frontend


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ItchIdentifier")


class ItchIdentifier:

    CONFIDENCE_THRESHOLD = 0.95
    DETECTION_THRESHOLD = 0.10

    DANGEROUS_THRESHOLD = 10
    PROXIMITY_THRESHOLD = 0.3048

    SPECIES_CLASSIFICATION = {
        "Giant Hogweed": "dangerous",
        "Poison Hemlock": "dangerous",
        "Spotted Water Hemlock": "dangerous",
        "Mayapple": "dangerous",
    }

    def __init__(
        self,
        model_path="models/best.pt",
        device="cpu",
        distance_estimator=None,
    ):

        self.model_path = Path(model_path)
        self.device = device
        self.distance_estimator = distance_estimator

        self.paused = False
        self.frame_number = 0

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}"
            )

        self.detection_model = YOLO(
            str(self.model_path)
        )

        self.classification_model = (
            self.create_classification_model()
        )

        logger.info(
            "Itch Identifier initialized successfully."
        )

    # --------------------------------------------------
    # MODEL TRAINING
    # --------------------------------------------------

    @staticmethod
    def create_detection_model(
        dataset,
        epochs=100,
        image_size=640,
        batch_size=16,
        output_dir="runs/itch_identifier",
        device="cpu",
    ):

        dataset_path = Path(dataset).resolve()
        output_path = Path(output_dir).resolve()

        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset configuration not found: {dataset_path}"
            )

        if epochs <= 0 or batch_size <= 0:
            raise ValueError(
                "Epochs and batch size must be positive."
            )

        if image_size <= 0:
            raise ValueError(
                "Image size must be positive."
            )

        model = YOLO("yolo11n.pt")

        model.train(
            data=str(dataset_path),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=str(output_path),
            name="species_detection",
            device=device,
            patience=20,
            save=True,
        )

        best_weights = (
            output_path
            / "species_detection"
            / "weights"
            / "best.pt"
        )

        if not best_weights.is_file():
            raise FileNotFoundError(
                f"Training weights not found: {best_weights}"
            )

        trained_model = YOLO(str(best_weights))

        metrics = trained_model.val(
            data=str(dataset_path),
            imgsz=image_size,
            device=device,
        )

        logger.info("Model training completed.")

        return trained_model, metrics

    # --------------------------------------------------
    # SPECIES CLASSIFICATION
    # --------------------------------------------------

    def create_classification_model(self):

        return self.SPECIES_CLASSIFICATION.copy()

    @staticmethod
    def normalize_species_name(species):

        return (
            str(species)
            .split(" -", 1)[0]
            .strip()
        )

    def label_function(self, species):

        normalized = self.normalize_species_name(
            species
        )

        return self.classification_model.get(
            normalized,
            "uncertain"
        )

    # --------------------------------------------------
    # PLANT DETECTION
    # --------------------------------------------------

    def annotate(
        self,
        frame,
        detection_model=None,
        classification_model=None,
        label_function=None,
    ):

        if frame is None:
            raise ValueError(
                "Input frame cannot be None."
            )

        model = (
            detection_model
            if detection_model is not None
            else self.detection_model
        )

        classifications = (
            classification_model
            if classification_model is not None
            else self.classification_model
        )

        if label_function is not None:
            classify = label_function
        else:
            classify = lambda species: classifications.get(
                self.normalize_species_name(species),
                "uncertain"
            )

        results = model.predict(
            source=frame,
            conf=self.DETECTION_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        detections = []

        if not results:
            return frame, detections

        result = results[0]

        if result.boxes is None:
            return frame, detections

        for box in result.boxes:

            confidence = float(
                box.conf[0].item()
            )

            class_id = int(
                box.cls[0].item()
            )

            species = result.names[class_id]

            coordinates = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coordinates.tolist()

            if confidence < self.CONFIDENCE_THRESHOLD:
                hazard = "uncertain"
            else:
                hazard = classify(species)

            if hazard not in (
                "dangerous",
                "non-dangerous",
                "uncertain",
            ):
                hazard = "uncertain"

            detections.append({
                "species": str(species),
                "confidence": confidence,
                "hazard": hazard,
                "bbox": (x1, y1, x2, y2),
                "distance_m": None,
            })

        return frame, detections

    # --------------------------------------------------
    # DISTANCE ESTIMATION
    # --------------------------------------------------

    def distance(self, frame, detections):

        if self.distance_estimator is None:
            return detections

        for plant in detections:

            if plant.get("hazard") != "dangerous":
                plant["distance_m"] = None
                continue

            try:

                estimator = self.distance_estimator

                if hasattr(estimator, "estimate"):
                    value = estimator.estimate(
                        plant["bbox"]
                    )
                elif callable(estimator):
                    value = estimator(
                        plant["bbox"]
                    )
                else:
                    raise TypeError(
                        "Distance estimator must implement "
                        "estimate(bbox) or be callable."
                    )

                if value is None:
                    plant["distance_m"] = None
                    continue

                value = float(value)

                if not math.isfinite(value) or value < 0:
                    plant["distance_m"] = None
                    continue

                plant["distance_m"] = value

            except Exception:

                plant["distance_m"] = None

                logger.exception(
                    "Distance estimation failed."
                )

        return detections

    # --------------------------------------------------
    # DANGEROUS PLANT COUNTING
    # --------------------------------------------------

    @staticmethod
    def count_dangerous_plants(detections):

        return sum(
            1
            for detection in detections
            if detection.get("hazard") == "dangerous"
        )

    # --------------------------------------------------
    # AREA SAFETY SIGNAL
    # --------------------------------------------------

    def area_signal(
        self,
        frame,
        detections,
        count_threshold=None,
    ):

        if count_threshold is None:
            count_threshold = self.DANGEROUS_THRESHOLD

        dangerous_count = self.count_dangerous_plants(
            detections
        )

        unsafe_area = (
            dangerous_count > count_threshold
        )

        if unsafe_area:
            logger.warning(
                "UNSAFE AREA: %s dangerous plants detected.",
                dangerous_count
            )

        return frame, unsafe_area, dangerous_count

    # --------------------------------------------------
    # PROXIMITY ALERT
    # --------------------------------------------------

    def proximity_alert(self, frame, detections):

        close_plants = []

        for plant in detections:

            if plant.get("hazard") != "dangerous":
                continue

            distance_m = plant.get("distance_m")

            if distance_m is None:
                continue

            try:
                distance_m = float(distance_m)
            except (TypeError, ValueError):
                continue

            if (
                math.isfinite(distance_m)
                and 0 <= distance_m < self.PROXIMITY_THRESHOLD
            ):
                close_plants.append(plant)

        if close_plants:
            logger.warning(
                "%s dangerous plant(s) within one foot.",
                len(close_plants)
            )

        return frame, close_plants

    # --------------------------------------------------
    # VIDEO CAPTURE
    # --------------------------------------------------

    def capture(
        self,
        video_feed_path=0,
        time_slow_down=0.01,
        display_video=True,
    ):

        if time_slow_down < 0:
            raise ValueError(
                "time_slow_down cannot be negative."
            )

        cap = cv2.VideoCapture(video_feed_path)

        if not cap.isOpened():
            cap.release()

            raise RuntimeError(
                f"Could not open video: {video_feed_path}"
            )

        self.paused = False
        self.frame_number = 0

        logger.info("Video capture started.")

        try:

            interface = frontend.initialize_interface()

            while True:

                # PAUSE / RESUME

                if self.paused:

                    if display_video:

                        key = cv2.waitKey(100) & 0xFF

                        if key == ord("p"):
                            self.pause()

                        elif key == ord("q"):
                            break

                    else:
                        time.sleep(0.05)

                    continue

                # READ FRAME

                success, frame = cap.read()

                if not success:
                    logger.info("Video stream ended.")
                    break

                self.frame_number += 1

                # DETECTION

                _, detections = self.annotate(frame)

                detections = self.distance(
                    frame,
                    detections
                )

                # SAFETY CALCULATIONS

                _, unsafe_area, dangerous_count = (
                    self.area_signal(
                        frame,
                        detections
                    )
                )

                _, close_plants = self.proximity_alert(
                    frame,
                    detections
                )

                # FRONTEND DATA CONTRACT

                interface["frame"] = frame
                interface["detections"] = detections
                interface["unsafe_area"] = unsafe_area
                interface["dangerous_count"] = dangerous_count
                interface["dangerous_threshold"] = (
                    self.DANGEROUS_THRESHOLD
                )
                interface["close_plants"] = close_plants

                interface = frontend.count_plants(
                    interface
                )

                interface = frontend.set_status(
                    interface
                )

                interface = frontend.mark_up(
                    interface
                )

                # DISPLAY

                annotated_frame = interface[
                    "annotated_frame"
                ]

                if display_video:

                    cv2.imshow(
                        "Itch Identifier",
                        annotated_frame
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("p"):
                        self.pause()

                    elif key == ord("q"):
                        break

                if time_slow_down:
                    time.sleep(time_slow_down)

        except KeyboardInterrupt:

            logger.info("Application interrupted.")

        finally:

            cap.release()

            if display_video:
                cv2.destroyAllWindows()

            logger.info(
                "Video capture stopped. Frames processed: %s",
                self.frame_number
            )

    # --------------------------------------------------
    # PAUSE AND RESUME
    # --------------------------------------------------

    def pause(self):

        self.paused = not self.paused

        if self.paused:
            logger.info("Application paused.")
        else:
            logger.info("Application resumed.")
