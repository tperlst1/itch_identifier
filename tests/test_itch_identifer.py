
import numpy as np
import pytest

from src.itch_identifier import ItchIdentifier
from src.distance_estimator import DistanceEstimator
from src import frontend


def make_app():

    app = ItchIdentifier.__new__(ItchIdentifier)

    app.distance_estimator = None
    app.device = "cpu"
    app.paused = False
    app.frame_number = 0

    app.classification_model = (
        ItchIdentifier.SPECIES_CLASSIFICATION.copy()
    )

    return app


def make_detection(
    hazard="dangerous",
    distance=None
):

    return {
        "species": "Giant Hogweed",
        "confidence": 0.98,
        "hazard": hazard,
        "bbox": (10, 10, 100, 100),
        "distance_m": distance,
    }


# ----------------------------------------------
# CLASSIFICATION TESTS
# ----------------------------------------------

def test_common_species_names():

    app = make_app()

    assert app.label_function(
        "Giant Hogweed"
    ) == "dangerous"

    assert app.label_function(
        "Mayapple"
    ) == "dangerous"

    assert app.label_function(
        "Unknown Plant"
    ) == "uncertain"


def test_roboflow_species_names():

    app = make_app()

    assert app.label_function(
        "Giant Hogweed -Heracleum mantegazzianum-"
    ) == "dangerous"

    assert app.label_function(
        "Mayapple -Podophyllum peltatum-"
    ) == "dangerous"

    assert app.label_function(
        "Poison Hemlock -Conium maculatum-"
    ) == "dangerous"

    assert app.label_function(
        "Spotted Water Hemlock -Cicuta maculata-"
    ) == "dangerous"


def test_unknown_species():

    app = make_app()

    assert app.label_function(
        "Unrecognized Plant"
    ) == "uncertain"


# ----------------------------------------------
# DETECTION COUNT TESTS
# ----------------------------------------------

def test_dangerous_plant_count():

    detections = [
        make_detection(),
        make_detection(),
        make_detection("uncertain"),
    ]

    assert ItchIdentifier.count_dangerous_plants(
        detections
    ) == 2


def test_area_threshold_exceeded():

    app = make_app()

    detections = [
        make_detection()
        for _ in range(11)
    ]

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    _, unsafe, count = app.area_signal(
        frame,
        detections
    )

    assert unsafe is True
    assert count == 11


def test_area_threshold_not_exceeded():

    app = make_app()

    detections = [
        make_detection()
        for _ in range(10)
    ]

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    _, unsafe, count = app.area_signal(
        frame,
        detections
    )

    assert unsafe is False
    assert count == 10


# ----------------------------------------------
# DISTANCE TESTS
# ----------------------------------------------

def test_distance_without_calibration():

    app = make_app()

    detections = [
        make_detection()
    ]

    result = app.distance(
        None,
        detections
    )

    assert result[0]["distance_m"] is None


def test_distance_with_mock_estimator():

    app = make_app()

    class MockEstimator:

        def estimate(self, bbox):
            return 0.2

    app.distance_estimator = MockEstimator()

    detections = [
        make_detection()
    ]

    result = app.distance(
        None,
        detections
    )

    assert result[0]["distance_m"] == 0.2


def test_distance_with_callable_estimator():

    app = make_app()

    app.distance_estimator = lambda bbox: 0.5

    detections = [
        make_detection()
    ]

    result = app.distance(
        None,
        detections
    )

    assert result[0]["distance_m"] == 0.5


def test_distance_estimator_failure():

    app = make_app()

    class BrokenEstimator:

        def estimate(self, bbox):
            raise RuntimeError("Calibration failure")

    app.distance_estimator = BrokenEstimator()

    detections = [
        make_detection()
    ]

    result = app.distance(
        None,
        detections
    )

    assert result[0]["distance_m"] is None


def test_invalid_distance_is_rejected():

    app = make_app()

    class InvalidEstimator:

        def estimate(self, bbox):
            return float("nan")

    app.distance_estimator = InvalidEstimator()

    detections = [
        make_detection()
    ]

    result = app.distance(
        None,
        detections
    )

    assert result[0]["distance_m"] is None


def test_non_dangerous_distance_not_estimated():

    app = make_app()

    class MockEstimator:

        def estimate(self, bbox):
            return 0.2

    app.distance_estimator = MockEstimator()

    detections = [
        make_detection("uncertain")
    ]

    app.distance(None, detections)

    assert detections[0]["distance_m"] is None


def test_proximity_alert():

    app = make_app()

    detections = [
        make_detection(distance=0.2),
        make_detection(distance=2.0),
    ]

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    _, close_plants = app.proximity_alert(
        frame,
        detections
    )

    assert len(close_plants) == 1


# ----------------------------------------------
# DISTANCE ESTIMATOR TESTS
# ----------------------------------------------

def test_invalid_homography_shape(tmp_path):

    calibration_file = (
        tmp_path / "invalid_calibration.npz"
    )

    np.savez(
        calibration_file,
        homography=np.eye(2)
    )

    with pytest.raises(ValueError):
        DistanceEstimator(calibration_file)


def test_valid_homography(tmp_path):

    calibration_file = (
        tmp_path / "valid_calibration.npz"
    )

    np.savez(
        calibration_file,
        homography=np.eye(3)
    )

    estimator = DistanceEstimator(
        calibration_file
    )

    result = estimator.estimate(
        (10, 10, 100, 100)
    )

    assert result is not None
    assert result >= 0


def test_invalid_bounding_box(tmp_path):

    calibration_file = (
        tmp_path / "valid_calibration.npz"
    )

    np.savez(
        calibration_file,
        homography=np.eye(3)
    )

    estimator = DistanceEstimator(
        calibration_file
    )

    assert estimator.estimate(
        (100, 100, 10, 10)
    ) is None


# ----------------------------------------------
# FRONTEND INTEGRATION TESTS
# ----------------------------------------------

def test_frontend_initialization():

    interface = frontend.initialize_interface()

    assert interface["frame"] is None
    assert interface["detections"] == []
    assert interface["dangerous_count"] == 0


def test_frontend_count():

    interface = frontend.initialize_interface()

    interface["detections"] = [
        make_detection(),
        make_detection(),
    ]

    interface["dangerous_count"] = 2

    result = frontend.count_plants(interface)

    assert result["dangerous_count"] == 2


def test_frontend_integration():

    interface = frontend.initialize_interface()

    interface["frame"] = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    interface["detections"] = [
        make_detection(distance=0.2)
    ]

    interface["unsafe_area"] = False
    interface["dangerous_count"] = 1

    interface["close_plants"] = [
        interface["detections"][0]
    ]

    interface = frontend.count_plants(interface)
    interface = frontend.set_status(interface)
    interface = frontend.mark_up(interface)

    assert interface["dangerous_count"] == 1

    assert (
        interface["status_text"]
        == "DANGER: TOO CLOSE - MOVE AWAY"
    )

    assert interface["annotated_frame"].shape == (
        480, 640, 3
    )


def test_frontend_handles_invalid_distance():

    interface = frontend.initialize_interface()

    interface["frame"] = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    interface["detections"] = [
        make_detection(distance=float("nan"))
    ]

    interface = frontend.count_plants(interface)
    interface = frontend.set_status(interface)
    interface = frontend.mark_up(interface)

    assert interface["annotated_frame"] is not None


def test_frontend_missing_frame():

    interface = frontend.initialize_interface()

    with pytest.raises(ValueError):
        frontend.mark_up(interface)
