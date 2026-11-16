"""
Itch Identifier - Unit Tests

Run:
    pytest tests/ -v
"""

from src.itch_identifier import ItchIdentifier


def create_application():

    application = ItchIdentifier.__new__(
        ItchIdentifier
    )

    application.classification_model = (
        ItchIdentifier.SPECIES_CLASSIFICATION.copy()
    )

    return application


def test_dangerous_species():

    application = create_application()

    assert application.label_function(
        "Giant Hogweed"
    ) == "dangerous"


def test_unknown_species():

    application = create_application()

    assert application.label_function(
        "Unknown Plant"
    ) == "uncertain"


def test_dangerous_plant_count():

    detections = [
        {"hazard": "dangerous"}
        for _ in range(11)
    ]

    assert (
        ItchIdentifier.count_dangerous_plants(detections)
        == 11
    )


def test_non_dangerous_plants_not_counted():

    detections = [
        {"hazard": "dangerous"},
        {"hazard": "non-dangerous"},
        {"hazard": "uncertain"},
    ]

    assert (
        ItchIdentifier.count_dangerous_plants(detections)
        == 1
    )


def test_area_threshold():

    detections = [
        {"hazard": "dangerous"}
        for _ in range(11)
    ]

    assert (
        ItchIdentifier.count_dangerous_plants(detections)
        > ItchIdentifier.DANGEROUS_THRESHOLD
    )


def test_area_threshold_not_exceeded():

    detections = [
        {"hazard": "dangerous"}
        for _ in range(10)
    ]

    assert (
        ItchIdentifier.count_dangerous_plants(detections)
        == ItchIdentifier.DANGEROUS_THRESHOLD
    )


def test_proximity_threshold():

    assert ItchIdentifier.PROXIMITY_THRESHOLD == 0.3048


def test_distance_below_one_foot():

    distance = 0.20

    assert distance < ItchIdentifier.PROXIMITY_THRESHOLD


def test_distance_above_one_foot():

    distance = 0.50

    assert distance > ItchIdentifier.PROXIMITY_THRESHOLD
