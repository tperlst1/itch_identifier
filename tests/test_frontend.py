
import cv2
import numpy as np

from src import frontend


def create_test_interface():

    interface = frontend.initialize_interface()

    interface["frame"] = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    interface["detections"] = [
        {
            "species": "Giant Hogweed",
            "hazard": "dangerous",
            "confidence": 0.98,
            "bbox": (100, 100, 250, 250),
            "distance_m": 0.25
        },
        {
            "species": "Clover",
            "hazard": "non-dangerous",
            "confidence": 0.97,
            "bbox": (300, 100, 400, 250),
            "distance_m": None
        },
        {
            "species": "Unknown Plant",
            "hazard": "uncertain",
            "confidence": 0.72,
            "bbox": (450, 100, 550, 250),
            "distance_m": None
        }
    ]

    return interface


def test_count_plants():

    interface = create_test_interface()

    interface = frontend.count_plants(interface)

    assert interface["dangerous_count"] == 1


def test_count_more_than_ten_dangerous_plants():

    interface = create_test_interface()

    interface["detections"] = [
        {
            "hazard": "dangerous"
        }
        for _ in range(11)
    ]

    interface = frontend.count_plants(interface)

    assert interface["dangerous_count"] == 11

    assert "AREA UNSAFE" in interface["plant_count_text"]


def test_set_status_proximity_priority():

    interface = create_test_interface()

    interface["unsafe_area"] = True

    interface["close_plants"] = [
        {"species": "Giant Hogweed"}
    ]

    interface = frontend.set_status(interface)

    assert "TOO CLOSE" in interface["status_text"]


def test_set_status_area_warning():

    interface = create_test_interface()

    interface["unsafe_area"] = True

    interface = frontend.set_status(interface)

    assert "UNSAFE AREA" in interface["status_text"]


def test_set_status_uncertain():

    interface = create_test_interface()

    interface = frontend.set_status(interface)

    assert "UNIDENTIFIED" in interface["status_text"]


def test_mark_up():

    interface = create_test_interface()

    interface = frontend.mark_up(interface)

    assert interface["annotated_frame"] is not None

    assert interface["annotated_frame"].shape == (
        480, 640, 3
    )


def test_mark_up_does_not_modify_original_frame():

    interface = create_test_interface()

    original = interface["frame"].copy()

    interface = frontend.mark_up(interface)

    assert np.array_equal(
        original,
        interface["frame"]
    )


def test_missing_frame():

    interface = frontend.initialize_interface()

    try:

        frontend.mark_up(interface)

        assert False, "Expected ValueError"

    except ValueError:

        assert True
