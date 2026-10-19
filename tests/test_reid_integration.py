"""Edge-case tests for traffic_analysis.reid.CanonicalCounter."""

from __future__ import annotations

import numpy as np

from traffic_analysis.reid import (
    CanonicalCounter,
    appearance_similarity,
    extract_appearance,
    REID_HIST_THRESHOLD,
)


def _frame_with_color(color_bgr, size=(120, 90, 3)):
    return np.full(size, color_bgr, dtype=np.uint8)


def _bbox(cx, cy, w=40, h=60):
    return (
        cx - w // 2,
        cy - h // 2,
        cx + w // 2,
        cy + h // 2,
    )


def _obs(track_id, class_id, bbox, centroid):
    return (track_id, class_id, bbox, centroid)


def test_frame_none_does_not_merge_new_track_without_appearance():
    counter = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))

    counter.register_frame(
        [_obs(1, 2, _bbox(50, 50), (50.0, 50.0))],
        1,
        frame,
    )
    original_id = counter.get_canonical_id(1)

    results = counter.register_frame(
        [_obs(2, 2, _bbox(52, 52), (52.0, 52.0))],
        2,
        None,
    )

    assert results[2] != original_id
    assert counter.get_unique_count() == 2


def test_degenerate_out_of_frame_crop_does_not_merge():
    counter = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))

    counter.register_frame(
        [_obs(1, 2, _bbox(50, 50), (50.0, 50.0))],
        1,
        frame,
    )
    original_id = counter.get_canonical_id(1)

    results = counter.register_frame(
        [_obs(2, 2, (-100, -100, -50, -50), (50.0, 50.0))],
        2,
        frame,
    )

    assert results[2] != original_id
    assert counter.get_unique_count() == 2


def test_black_crop_has_zero_histogram():
    black = _frame_with_color((0, 0, 0))
    colored = _frame_with_color((30, 100, 200))

    zero_appearance = extract_appearance(black, _bbox(50, 50), 2, 1)
    real_appearance = extract_appearance(colored, _bbox(50, 50), 2, 2)

    # Black is a valid appearance sample, not a missing/invalid crop.
    assert np.all(np.isfinite(zero_appearance.hist))
    assert np.isclose(np.linalg.norm(zero_appearance.hist), 1.0)
    assert (
        appearance_similarity(zero_appearance, real_appearance)
        < REID_HIST_THRESHOLD
    )


def test_no_id_detection_is_filtered_before_counter_registration():
    """Only valid tracker IDs should be sent to CanonicalCounter."""
    counter = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))

    detections = [
        _obs(5, 2, _bbox(50, 50), (50.0, 50.0)),
        _obs(None, 2, _bbox(200, 200), (200.0, 200.0)),
    ]

    valid_observations = [
        observation
        for observation in detections
        if observation[0] is not None
    ]

    counter.register_frame(valid_observations, 1, frame)

    assert counter.get_unique_count() == 1
    assert counter.get_canonical_id(5) is not None


def test_match_decisions_expose_score_and_rejection_reason():
    counter = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))

    first = counter.register_frame(
        [_obs(1, 2, _bbox(50, 50), (50.0, 50.0))],
        1,
        frame,
    )
    first_decision = counter.get_frame_diagnostics()[0]
    assert first_decision.canonical_id == first[1]
    assert first_decision.matched_existing is False
    assert first_decision.reason == "no_class_compatible_candidate"

    second = counter.register_frame(
        [_obs(2, 2, _bbox(51, 50), (51.0, 50.0))],
        2,
        frame,
    )
    second_decision = counter.get_frame_diagnostics()[0]
    assert second[2] == first[1]
    assert second_decision.matched_existing is True
    assert second_decision.similarity == 1.0
    assert second_decision.reason == "class_motion_appearance_match"


def test_invalid_crop_match_is_rejected_with_diagnostic():
    counter = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    counter.register_frame(
        [_obs(1, 2, _bbox(50, 50), (50.0, 50.0))],
        1,
        frame,
    )

    result = counter.register_frame(
        [_obs(2, 2, (-100, -100, -50, -50), (50.0, 50.0))],
        2,
        frame,
    )

    decision = counter.get_frame_diagnostics()[0]
    assert result[2] != counter.get_canonical_id(1)
    assert decision.matched_existing is False
    assert decision.similarity is None
    assert decision.reason == "invalid_appearance"
