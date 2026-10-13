from traffic_analysis.class_resolution import (
    UNCERTAIN_CLASS,
    resolve_track_class,
)


NAMES = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}


def test_stable_class_evidence_resolves():
    observations = [(1, 3, 0.8), (2, 3, 0.9), (3, 3, 0.7)]

    assert resolve_track_class(observations, 4, NAMES) == "Bike"


def test_single_class_flip_does_not_override_strong_prior_evidence():
    observations = [
        (1, 2, 0.9), (2, 2, 0.9), (3, 2, 0.9), (4, 3, 0.9),
    ]

    assert resolve_track_class(observations, 5, NAMES) == "Car"


def test_fewer_than_three_eligible_observations_is_uncertain():
    observations = [(1, 2, 0.9), (2, 2, 0.9)]

    assert resolve_track_class(observations, 3, NAMES) == UNCERTAIN_CLASS


def test_conflicting_evidence_below_seventy_percent_is_uncertain():
    observations = [(1, 2, 0.8), (2, 2, 0.8), (3, 3, 0.9)]

    assert resolve_track_class(observations, 4, NAMES) == UNCERTAIN_CLASS


def test_low_confidence_observations_are_ignored():
    observations = [
        (1, 3, 0.9), (2, 3, 0.9), (3, 3, 0.9),
        (4, 2, 0.39), (5, 2, 0.2),
    ]

    assert resolve_track_class(observations, 6, NAMES) == "Bike"


def test_confidence_threshold_is_inclusive():
    observations = [(1, 3, 0.4), (2, 3, 0.4), (3, 3, 0.4)]

    assert resolve_track_class(observations, 4, NAMES) == "Bike"


def test_support_threshold_is_inclusive():
    observations = [(1, 2, 0.7), (2, 2, 0.7), (3, 3, 0.6)]

    assert resolve_track_class(observations, 4, NAMES) == "Car"


def test_crossing_and_future_observations_are_excluded():
    observations = [
        (1, 3, 0.9), (2, 3, 0.9), (3, 3, 0.9),
        (4, 2, 1.0), (5, 2, 1.0),
    ]

    assert resolve_track_class(observations, 4, NAMES) == "Bike"


def test_invalid_class_ids_are_ignored():
    observations = [(1, 99, 0.9), (2, 99, 0.9), (3, 99, 0.9)]

    assert resolve_track_class(observations, 4, NAMES) == UNCERTAIN_CLASS


def test_empty_history_is_uncertain():
    assert resolve_track_class([], 4, NAMES) == UNCERTAIN_CLASS