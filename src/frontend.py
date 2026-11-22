
"""
Itch Identifier - Frontend Module

Responsibilities:
    1. Display current safety status.
    2. Display dangerous plant count.
    3. Annotate detected plants with high-contrast labels.

Backend integration:
    ItchIdentifier.annotate()
    ItchIdentifier.area_signal()
    ItchIdentifier.proximity_alert()
"""

import cv2


# --------------------------------------------------
# DISPLAY CONFIGURATION
# --------------------------------------------------

COLORS = {
    "dangerous": (0, 0, 255),        # Red
    "non-dangerous": (0, 200, 0),   # Green
    "uncertain": (0, 165, 255),     # Orange
}

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# --------------------------------------------------
# INITIALIZE FRONTEND
# --------------------------------------------------

def initialize_interface():

    return {
        "frame": None,
        "detections": [],
        "unsafe_area": False,
        "dangerous_count": 0,
        "close_plants": [],
        "status_text": "",
        "status_color": WHITE,
        "plant_count_text": "",
        "annotated_frame": None,
    }


# --------------------------------------------------
# SET USER SAFETY STATUS
# --------------------------------------------------

def set_status(interface):
    """
    Generate the user's current safety status.

    Priority:
        1. Dangerous plant within one foot.
        2. More than 10 dangerous plants.
        3. Uncertain plant detections.
        4. No identified danger thresholds exceeded.
    """

    close_plants = interface.get("close_plants", [])
    unsafe_area = interface.get("unsafe_area", False)

    detections = interface.get("detections", [])

    uncertain_detected = any(
        plant.get("hazard", "uncertain").lower() == "uncertain"
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
# COUNT DANGEROUS PLANTS
# --------------------------------------------------

def count_plants(interface):
    """
    Count dangerous plants detected in the current frame.
    """

    detections = interface.get("detections", [])

    dangerous_count = sum(
        1
        for plant in detections
        if plant.get("hazard", "uncertain").lower() == "dangerous"
    )

    interface["dangerous_count"] = dangerous_count

    if dangerous_count > 10:

        count_text = (
            f"DANGEROUS PLANTS: {dangerous_count} "
            "- AREA UNSAFE"
        )

    else:

        count_text = (
            f"DANGEROUS PLANTS DETECTED: {dangerous_count}"
        )

    interface["plant_count_text"] = count_text

    return interface


# --------------------------------------------------
# MARK UP DETECTED PLANTS
# --------------------------------------------------

def mark_up(interface):
    """
    Draw high-contrast bounding boxes and labels.

    Labels include:
        - Plant species
        - Hazard classification
        - Detection confidence
        - Distance in feet, when available
    """

    frame = interface.get("frame")

    if frame is None:
        raise ValueError("No video frame supplied to frontend.")

    detections = interface.get("detections", [])

    annotated_frame = frame.copy()

    height, width = annotated_frame.shape[:2]

    for plant in detections:

        species = plant.get("species", "Unknown species")

        hazard = plant.get(
            "hazard", "uncertain"
        ).lower()

        confidence = float(
            plant.get("confidence", 0.0)
        )

        bbox = plant.get("bbox")

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Keep coordinates within frame boundaries.
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = COLORS.get(
            hazard, COLORS["uncertain"]
        )

        # Draw bounding box.
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # Generate plant identification label.
        label = (
            f"{species} | {hazard.upper()} "
            f"| {confidence:.1%}"
        )

        # Include distance when available.
        distance_m = plant.get("distance_m")

        if distance_m is not None:
            label += f" | {distance_m * 3.28084:.2f} ft"
        else:
            label += " | DISTANCE UNKNOWN"

        # Calculate label dimensions.
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
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

        # Draw colored label background.
        cv2.rectangle(
            annotated_frame,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            -1
        )

        # Draw white label text.
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, label_top + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------
    # DISPLAY SAFETY STATUS
    # --------------------------------------------------

    status = interface.get("status_text", "")

    status_color = interface.get(
        "status_color", WHITE
    )

    if status:

        cv2.rectangle(
            annotated_frame,
            (0, 0),
            (width, 55),
            BLACK,
            -1
        )

        cv2.putText(
            annotated_frame,
            status,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------
    # DISPLAY DANGEROUS PLANT COUNT
    # --------------------------------------------------

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
            cv2.LINE_AA
        )

    interface["annotated_frame"] = annotated_frame

    return interface
