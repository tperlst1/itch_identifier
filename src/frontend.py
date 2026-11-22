
"""
Itch Identifier - Frontend

Responsibilities:
- Render detection bounding boxes.
- Display species and confidence.
- Display optional distance measurements.
- Display dangerous plant count.
- Display safety status.

All detection and classification decisions
are made by the backend.
"""

import cv2
import math


COLORS = {
    "dangerous": (0, 0, 255),
    "non-dangerous": (0, 200, 0),
    "uncertain": (0, 165, 255),
}

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# --------------------------------------------------
# INITIALIZE INTERFACE
# --------------------------------------------------

def initialize_interface():

    return {
        "frame": None,
        "detections": [],
        "unsafe_area": False,
        "dangerous_count": 0,
        "dangerous_threshold": 10,
        "close_plants": [],
        "status_text": "",
        "status_color": WHITE,
        "plant_count_text": "",
        "annotated_frame": None,
    }


# --------------------------------------------------
# COUNT DANGEROUS PLANTS
# --------------------------------------------------

def count_plants(interface):

    detections = interface.get("detections", [])

    dangerous_count = sum(
        1
        for plant in detections
        if plant.get("hazard", "uncertain") == "dangerous"
    )

    backend_count = interface.get("dangerous_count")

    if (
        backend_count is not None
        and backend_count != dangerous_count
    ):
        raise ValueError(
            "Backend/frontend dangerous plant count mismatch."
        )

    interface["dangerous_count"] = dangerous_count

    threshold = interface.get(
        "dangerous_threshold", 10
    )

    if dangerous_count > threshold:

        interface["plant_count_text"] = (
            f"DANGEROUS PLANTS: {dangerous_count} "
            "- AREA UNSAFE"
        )

    else:

        interface["plant_count_text"] = (
            f"DANGEROUS PLANTS DETECTED: {dangerous_count}"
        )

    return interface


# --------------------------------------------------
# GENERATE SAFETY STATUS
# --------------------------------------------------

def set_status(interface):

    detections = interface.get("detections", [])
    close_plants = interface.get("close_plants", [])
    unsafe_area = interface.get("unsafe_area", False)

    uncertain_detected = any(
        plant.get("hazard", "uncertain") == "uncertain"
        for plant in detections
    )

    if close_plants:

        status = "DANGER: TOO CLOSE - MOVE AWAY"
        color = COLORS["dangerous"]

    elif unsafe_area:

        status = "UNSAFE AREA - DO NOT PROCEED"
        color = COLORS["dangerous"]

    elif uncertain_detected:

        status = "CAUTION: UNIDENTIFIED PLANTS DETECTED"
        color = COLORS["uncertain"]

    else:

        status = "NO IDENTIFIED DANGER THRESHOLDS EXCEEDED"
        color = COLORS["non-dangerous"]

    interface["status_text"] = status
    interface["status_color"] = color

    return interface


# --------------------------------------------------
# SAFE DISTANCE FORMATTING
# --------------------------------------------------

def format_distance(distance_m):

    if distance_m is None:
        return "DISTANCE UNKNOWN"

    try:
        distance_m = float(distance_m)

    except (TypeError, ValueError):
        return "DISTANCE UNKNOWN"

    if not math.isfinite(distance_m) or distance_m < 0:
        return "DISTANCE UNKNOWN"

    distance_ft = distance_m * 3.28084

    return f"{distance_ft:.2f} ft"


# --------------------------------------------------
# RENDER DETECTIONS
# --------------------------------------------------

def mark_up(interface):

    frame = interface.get("frame")

    if frame is None:
        raise ValueError(
            "No video frame supplied to frontend."
        )

    annotated_frame = frame.copy()

    height, width = annotated_frame.shape[:2]

    detections = interface.get("detections", [])

    for plant in detections:

        species = plant.get(
            "species", "Unknown species"
        )

        hazard = plant.get(
            "hazard", "uncertain"
        )

        try:
            confidence = float(
                plant.get("confidence", 0.0)
            )

        except (TypeError, ValueError):
            confidence = 0.0

        if not math.isfinite(confidence):
            confidence = 0.0

        confidence = max(
            0.0,
            min(confidence, 1.0)
        )

        bbox = plant.get("bbox")

        if bbox is None or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = map(int, bbox)

        except (TypeError, ValueError):
            continue

        # Clamp coordinates to the frame.
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = COLORS.get(
            hazard,
            COLORS["uncertain"]
        )

        distance_label = format_distance(
            plant.get("distance_m")
        )

        label = (
            f"{species} | {hazard.upper()} "
            f"| {confidence:.1%} "
            f"| {distance_label}"
        )

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                2,
            )
        )

        label_top = max(
            0,
            y1 - text_height - baseline - 10
        )

        label_bottom = min(
            height - 1,
            label_top + text_height + baseline + 10
        )

        label_right = min(
            width - 1,
            x1 + text_width + 10
        )

        cv2.rectangle(
            annotated_frame,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, label_top + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            2,
            cv2.LINE_AA,
        )

    # ----------------------------------------------
    # SAFETY STATUS
    # ----------------------------------------------

    status = interface.get("status_text", "")

    status_color = interface.get(
        "status_color", WHITE
    )

    cv2.rectangle(
        annotated_frame,
        (0, 0),
        (width, 55),
        BLACK,
        -1,
    )

    if status:

        cv2.putText(
            annotated_frame,
            status,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA,
        )

    # ----------------------------------------------
    # DANGEROUS PLANT COUNT
    # ----------------------------------------------

    count_text = interface.get(
        "plant_count_text", ""
    )

    if count_text:

        cv2.putText(
            annotated_frame,
            count_text,
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA,
        )

    interface["annotated_frame"] = annotated_frame

    return interface
