"""Acceptance tests for separate unique-vehicle and line-crossing metrics."""

import pytest

from traffic_analysis import analytics
from traffic_analysis.reid import CanonicalCounter


def _summarize_metrics(*, unique_vehicles, crossing_events, untracked_detections):
    """Summarize already-resolved unique records, one per canonical ID."""
    canonical_ids = [vehicle["canonical_id"] for vehicle in unique_vehicles]
    assert len(canonical_ids) == len(set(canonical_ids)), (
        "unique_vehicles must contain exactly one resolved record per "
        "canonical vehicle ID"
    )

    summarize_metrics = getattr(analytics, "summarize_metrics", None)
    assert callable(summarize_metrics), (
        "traffic_analysis.analytics.summarize_metrics must expose the "
        "separate metric contract"
    )
    return summarize_metrics(
        unique_vehicles=unique_vehicles,
        crossing_events=crossing_events,
        untracked_detections=untracked_detections,
    )


def test_unique_vehicle_input_rejects_repeated_canonical_ids():
    with pytest.raises(
        AssertionError,
        match="one resolved record per canonical vehicle ID",
    ):
        _summarize_metrics(
            unique_vehicles=[
                {"canonical_id": 1, "vehicle_class": "Car"},
                {"canonical_id": 1, "vehicle_class": "Bike"},
            ],
            crossing_events=[],
            untracked_detections=0,
        )


def test_tracked_vehicle_without_crossing_counts_as_unique_only():
    metrics = _summarize_metrics(
        unique_vehicles=[{"canonical_id": 1, "vehicle_class": "Car"}],
        crossing_events=[],
        untracked_detections=0,
    )

    assert metrics["unique_vehicle_total"] == 1
    assert metrics["unique_vehicles_by_class"] == {
        "Car": 1,
        "Bike": 0,
        "Bus": 0,
        "Truck": 0,
    }
    assert metrics["line_crossings_total"] == 0
    assert metrics["line_crossings_by_class"] == {
        "Car": 0,
        "Bike": 0,
        "Bus": 0,
        "Truck": 0,
    }


def test_crossings_and_unique_vehicles_are_counted_separately_by_class():
    crossing_events = [
        {"event_type": "vehicle_crossing", "vehicle_class": "Bike"},
    ]
    metrics = _summarize_metrics(
        unique_vehicles=[
            {"canonical_id": 1, "vehicle_class": "Car"},
            {"canonical_id": 2, "vehicle_class": "Bike"},
        ],
        crossing_events=crossing_events,
        untracked_detections=0,
    )

    assert metrics["unique_vehicle_total"] == 2
    assert metrics["unique_vehicles_by_class"] == {
        "Car": 1,
        "Bike": 1,
        "Bus": 0,
        "Truck": 0,
    }
    assert metrics["line_crossings_total"] == 1
    assert metrics["line_crossings_by_class"] == {
        "Car": 0,
        "Bike": 1,
        "Bus": 0,
        "Truck": 0,
    }


def test_untracked_detections_are_diagnostic_not_confirmed_unique_vehicles():
    metrics = _summarize_metrics(
        unique_vehicles=[{"canonical_id": 1, "vehicle_class": "Car"}],
        crossing_events=[],
        untracked_detections=2,
    )

    assert metrics["untracked_detections"] == 2
    assert metrics["unique_vehicle_total"] == 1
    assert metrics["unique_vehicles_by_class"]["Car"] == 1
    assert metrics["line_crossings_total"] == 0


@pytest.mark.parametrize(
    ("detection_count", "tracked_detection_count", "expected"),
    [(3, 0, 3), (5, 3, 2), (4, 4, 0)],
)
def test_untracked_observations_are_detection_minus_tracked(
    detection_count,
    tracked_detection_count,
    expected,
):
    assert analytics.count_untracked_observations(
        detection_count, tracked_detection_count
    ) == expected


def test_canonical_records_feed_unique_and_crossing_metrics_separately():
    counter = CanonicalCounter(fps=25.0)
    counter.register_frame(
        [(5, 2, (0, 0, 10, 10), (5.0, 5.0))], 1, None
    )
    counter.register_frame(
        [(5, 3, (0, 0, 10, 10), (5.0, 5.0))], 2, None
    )
    vehicles = counter.get_canonical_vehicles({2: "Car", 3: "Bike"})
    events = [{"event_type": "vehicle_crossing", "vehicle_class": "Bike"}]

    metrics = analytics.summarize_metrics(
        unique_vehicles=vehicles,
        crossing_events=events,
        untracked_detections=2,
    )

    assert metrics["unique_vehicle_total"] == 1
    assert metrics["unique_vehicles_by_class"]["Car"] == 1
    assert metrics["unique_vehicles_by_class"]["Bike"] == 0
    assert metrics["line_crossings_total"] == 1
    assert metrics["line_crossings_by_class"]["Bike"] == 1
    assert metrics["untracked_detections"] == 2


def test_existing_event_analytics_remain_crossing_based():
    events = [
        {
            "event_type": "vehicle_crossing",
            "vehicle_class": "Bike",
            "canonical_id": 7,
            "session_id": "session-1",
        },
        {
            "event_type": "vehicle_crossing",
            "vehicle_class": "Bike",
            "canonical_id": 7,
            "session_id": "session-1",
        },
        {
            "event_type": "wrong_way",
            "vehicle_class": "Car",
            "session_id": "session-1",
        },
    ]

    summary = analytics.summarize_events(events)

    assert summary["total_crossings"] == 2
    assert summary["vehicle_counts"] == {
        "Car": 0,
        "Bike": 2,
        "Bus": 0,
        "Truck": 0,
    }
