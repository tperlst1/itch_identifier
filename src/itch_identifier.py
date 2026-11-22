
"""
Itch Identifier
===============

Main backend module.

Features:
1. YOLO-based plant species detection.
2. Manual dangerous/non-dangerous classification.
3. Uncertain identification below 95% confidence.
4. Plant species annotation.
5. Area warning when more than 10 dangerous plants
   are detected in a frame.
6. Proximity warning when a dangerous plant is
   less than one foot away.
7. OpenCV video capture and pause/resume controls.

Note:
Real-world distance estimation requires a calibrated
depth camera or equivalent validated depth source.
"""

import cv2
import time
import logging

from pathlib import Path
from typing import Callable, Optional

from ultralytics import YOLO

from src import frontend
from src.distance_estimator import DistanceEstimator


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ItchIdentifier")


# --------------------------------------------------
# MAIN APPLICATION CLASS
# --------------------------------------------------

class ItchIdentifier:

    CONFIDENCE_THRESHOLD = 0.95
    DETECTION_THRESHOLD = 0.10

    DANGEROUS_THRESHOLD = 10
    PROXIMITY_THRESHOLD = 0.3048  # One foot in meters

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
        distance_estimator: Optional[Callable] = None,
    ):

        self.model_path = Path(model_path)
        self.device = device
        self.distance_estimator = distance_estimator

        self.paused = False
        self.frame_number = 0

        if not self.model_path.exists():
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

        dataset_path = Path(dataset)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset configuration not found: {dataset}"
            )

        model = YOLO("yolo11n.pt")

        model.train(
            data=str(dataset_path),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=output_dir,
            name="species_detection",
            device=device,
            patience=20,
            save=True,
        )

        best_weights = (
            Path(output_dir)
            / "species_detection"
            / "weights"
            / "best.pt"
        )

        if not best_weights.exists():
            raise FileNotFoundError(
                "Training completed but best.pt was not found."
            )

        trained_model = YOLO(str(best_weights))

        metrics = trained_model.val(
            data=str(dataset_path),
            imgsz=image_size,
        )

        logger.info(
            "Itch Identifier model training completed."
        )

        return trained_model, metrics

    # --------------------------------------------------
    # MANUAL SPECIES CLASSIFICATION
    # --------------------------------------------------

    def create_classification_model(self):

        return self.SPECIES_CLASSIFICATION.copy()

    def label_function(self, species):

        return self.classification_model.get(
            species,
            "uncertain"
        )

    # --------------------------------------------------
    # PLANT DETECTION AND ANNOTATION
    # --------------------------------------------------

    def annotate(
        self,
        frame,
        detection_model=None,
        classification_model=None,
        label_function=None,
    ):

        detection_model = (
            detection_model or self.detection_model
        )

        classification_model = (
            classification_model or self.classification_model
        )

        if label_function is None:
            label_function = lambda species: (
                classification_model.get(
                    species,
                    "uncertain"
                )
            )

        results = detection_model.predict(
            source=frame,
            conf=self.DETECTION_THRESHOLD,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        for box in results.boxes:

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            species = results.names[class_id]

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coordinates.tolist()

            if confidence < self.CONFIDENCE_THRESHOLD:
                hazard = "uncertain"
            else:
                hazard = label_function(species)

            detection = {
                "species": species,
                "confidence": confidence,
                "hazard": hazard,
                "bbox": (x1, y1, x2, y2),
                "distance_m": None,
            }

            if self.distance_estimator is not None:

                try:

                    self.distance_estimator = DistanceEstimator(
                        "models/ground_calibration.npz"
                    )

                    if distance is not None and distance >= 0:
                        detection["distance_m"] = float(distance)

                except Exception as error:

                    logger.warning(
                        "Distance estimation failed: %s",
                        error
                    )

            detections.append(detection)

            self._draw_detection(
                frame,
                detection
            )

        return frame, detections

   def distance(self, frame, detections):
   
       for plant in detections:
   
           if plant.get("hazard") != "dangerous":
               plant["distance_m"] = None
               continue
   
           plant["distance_m"] = self.distance_estimator.estimate(
               plant["bbox"]
           )
          
   return detections
   
    # --------------------------------------------------
    # DRAW SPECIES AND HAZARD LABELS
    # --------------------------------------------------

    @staticmethod
    def _draw_detection(frame, detection):

        x1, y1, x2, y2 = detection["bbox"]

        hazard = detection["hazard"]

        colors = {
            "dangerous": (0, 0, 255),
            "non-dangerous": (0, 255, 0),
            "uncertain": (0, 165, 255),
        }

        color = colors.get(
            hazard,
            (255, 255, 255)
        )

        label = (
            f"{detection['species']} | "
            f"{hazard.upper()} | "
            f"{detection['confidence']:.1%}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2
        )

        if detection["distance_m"] is not None:

            distance_ft = (
                detection["distance_m"] * 3.28084
            )

            cv2.putText(
                frame,
                f"Distance: {distance_ft:.2f} ft",
                (x1, min(y2 + 20, frame.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # --------------------------------------------------
    # DANGEROUS PLANT COUNTING
    # --------------------------------------------------

    @staticmethod
    def count_dangerous_plants(detections):

        return sum(
            1
            for detection in detections
            if detection["hazard"] == "dangerous"
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

        unsafe_area = dangerous_count > count_threshold

        if unsafe_area:

            message = (
                f"UNSAFE AREA: {dangerous_count} "
                "DANGEROUS PLANTS DETECTED!"
            )

            cv2.putText(
                frame,
                message,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                3
            )

            logger.warning(message)

        return frame, unsafe_area, dangerous_count

    # --------------------------------------------------
    # PROXIMITY ALERT
    # --------------------------------------------------

   def proximity_alert(self, frame, detections):

    proximity_threshold = 0.3048  # One foot in meters

    close_plants = []

    for plant in detections:

        if plant.get("hazard") != "dangerous":
            continue

        # Retrieve distance calculated by distance()
        distance_m = plant.get("distance_m")

        if distance_m is None:
            continue

        if distance_m < proximity_threshold:
            close_plants.append(plant)

            cv2.putText(
                frame,
                "TOO CLOSE - MOVE AWAY",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
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
    """
    Capture video, process plant detections,
    evaluate safety conditions, and update
    the frontend display.
    """

    from src import frontend

    cap = cv2.VideoCapture(video_feed_path)

    if not cap.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not open video: {video_feed_path}"
        )

    self.paused = False
    self.frame_number = 0

    try:

        while True:

            if not self.paused:

                success, frame = cap.read()

                if not success:
                    break

                self.frame_number += 1

                
            # ------------------------------------------
            # BACKEND: DETECTION
            # ------------------------------------------
            
            # Preserve original frame for frontend rendering.
            raw_frame = frame.copy()
            
            # Run existing detection model.
            # Backend annotation is performed on a separate copy.
            annotated_frame, detections = self.annotate(frame.copy())

            detections = self.distance(frame, detections)
            
            
            # ------------------------------------------
            # BACKEND: SAFETY CALCULATIONS
            # ------------------------------------------
            
            # Determine dangerous plant count and area safety.
            _, unsafe, count = self.area_signal(
                frame.copy(),
                detections
            )
            
            # Determine dangerous plants within one foot.
            _, close_plants = self.proximity_alert(
                frame.copy(),
                detections
            )
            
            
            # ------------------------------------------
            # FRONTEND: INITIALIZE INTERFACE
            # ------------------------------------------
            
            interface = frontend.initialize_interface()
            
            interface["frame"] = raw_frame
            interface["detections"] = detections
            interface["unsafe_area"] = unsafe
            interface["dangerous_count"] = count
            interface["close_plants"] = close_plants
            
            
            # ------------------------------------------
            # FRONTEND: PROCESS DISPLAY INFORMATION
            # ------------------------------------------
            
            interface = frontend.count_plants(interface)
            
            interface = frontend.set_status(interface)
            
            interface = frontend.mark_up(interface)
            
            
            # ------------------------------------------
            # DISPLAY FINAL FRAME
            # ------------------------------------------
            
            annotated_frame = interface["annotated_frame"]
            
            # Retain existing frame information.
            cv2.putText(
                annotated_frame,
                (
                    f"Frame: {self.frame_number} "
                    f"| Objects: {len(detections)} "
                    f"| Dangerous: {count}"
                ),
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            
            if display_video:
            
                cv2.imshow(
                    "Itch Identifier",
                    annotated_frame
                )


            # --------------------------------------
            # KEYBOARD CONTROLS
            # --------------------------------------

            if display_video:

                key = cv2.waitKey(
                    1 if not self.paused else 100
                ) & 0xFF

                if key == ord("p"):

                    self.pause()

                elif key == ord("q"):

                    break

            if not self.paused:

                time.sleep(time_slow_down)

    except KeyboardInterrupt:

        logger.info("Itch Identifier stopped.")

    finally:

        cap.release()

        if display_video:

            cv2.destroyAllWindows()

        logger.info(
            "Itch Identifier processed %s frames.",
            self.frame_number
        )


        cap = cv2.VideoCapture(video_feed_path)

        if not cap.isOpened():

            cap.release()

            raise RuntimeError(
                f"Could not open video: {video_feed_path}"
            )

        self.paused = False
        self.frame_number = 0

        try:

            while True:

                if not self.paused:

                    success, frame = cap.read()

                    if not success:
                        break

                    self.frame_number += 1

                    annotated_frame, detections = (
                        self.annotate(frame)
                    )

                    annotated_frame, unsafe, count = (
                        self.area_signal(
                            annotated_frame,
                            detections
                        )
                    )

                    annotated_frame, close_plants = (
                        self.proximity_alert(
                            annotated_frame,
                            detections
                        )
                    )

                    cv2.putText(
                        annotated_frame,
                        (
                            f"Frame: {self.frame_number} "
                            f"| Objects: {len(detections)} "
                            f"| Dangerous: {count}"
                        ),
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    if display_video:

                        cv2.imshow(
                            "Itch Identifier",
                            annotated_frame
                        )

                if display_video:

                    key = cv2.waitKey(
                        1 if not self.paused else 100
                    ) & 0xFF

                    if key == ord("p"):
                        self.pause()

                    elif key == ord("q"):
                        break

                if not self.paused:
                    time.sleep(time_slow_down)

        except KeyboardInterrupt:

            logger.info("Itch Identifier stopped.")

        finally:

            cap.release()

            if display_video:
                cv2.destroyAllWindows()

            logger.info(
                "Itch Identifier processed %s frames.",
                self.frame_number
            )

    # --------------------------------------------------
    # PAUSE AND RESUME
    # --------------------------------------------------

    def pause(self):

        self.paused = not self.paused

        if self.paused:
            logger.info("Itch Identifier paused.")
        else:
            logger.info("Itch Identifier resumed.")
