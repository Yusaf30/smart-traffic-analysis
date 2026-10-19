"""Focused tests for traffic_analysis.reid.CanonicalCounter.

Tests use synthetic frames and pure-Python logic so they run without YOLO or
video files. Histogram-based appearance matching is exercised with solid-color
crops so similarity is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from traffic_analysis.reid import (
    CanonicalCounter,
    Appearance,
    CanonicalRecord,
    extract_appearance,
    appearance_similarity,
    class_compatible,
    motion_plausible,
    REID_HIST_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic frames
# ---------------------------------------------------------------------------

def _frame_with_color(color_bgr, size=(120, 90, 3)):
    return np.full(size, color_bgr, dtype=np.uint8)


# ---------------------------------------------------------------------------
# extract_appearance / similarity unit tests
# ---------------------------------------------------------------------------

def test_extract_appearance_same_color_high_similarity():
    frame = _frame_with_color((30, 100, 200))  # BGR
    a = extract_appearance(frame, _bbox(50, 50), 2, 1)
    b = extract_appearance(frame, _bbox(55, 55), 2, 2)
    sim = appearance_similarity(a, b)
    assert sim >= REID_HIST_THRESHOLD


def test_extract_appearance_different_color_low_similarity():
    frame1 = _frame_with_color((30, 100, 200))
    frame2 = _frame_with_color((200, 100, 30))
    a = extract_appearance(frame1, _bbox(50, 50), 2, 1)
    b = extract_appearance(frame2, _bbox(50, 50), 2, 2)
    sim = appearance_similarity(a, b)
    assert sim < REID_HIST_THRESHOLD


def test_spatial_histogram_rejects_same_colors_in_different_layouts():
    top = np.full((30, 40, 3), (0, 0, 255), dtype=np.uint8)
    bottom = np.full((30, 40, 3), (255, 0, 0), dtype=np.uint8)
    first = np.vstack((top, bottom))
    second = np.vstack((bottom, top))

    first_appearance = extract_appearance(first, (0, 0, 40, 60), 2, 1)
    second_appearance = extract_appearance(second, (0, 0, 40, 60), 2, 2)

    assert appearance_similarity(first_appearance, second_appearance) < REID_HIST_THRESHOLD


def test_class_compatible_zero_is_valid():
    # class_id 0 is a valid YOLO class; must not be treated as falsy.
    assert class_compatible(0, 0) is True
    assert class_compatible(0, 2) is False
    assert class_compatible(2, 2) is True


# ---------------------------------------------------------------------------
# CanonicalCounter: stable identity
# ---------------------------------------------------------------------------

def test_same_track_stays_same_canonical():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    results = []
    for f in (1, 2, 3):
        results.append(
            c.register_frame([(5, 2, _bbox(50, 50), (50.0, 50.0))], f, frame)
        )
    assert results[0][5] == results[1][5] == results[2][5]
    assert c.get_unique_count() == 1


def test_first_observation_is_added_once_to_class_history():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(5, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)

    rec = c._canonicals[c.get_canonical_id(5)]
    assert rec.class_history == [2]


def test_disabled_reid_does_not_merge_fragmented_track_ids():
    c = CanonicalCounter(enabled=False, fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(1, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    first_canonical = c.get_canonical_id(1)

    results = c.register_frame(
        [(2, 2, _bbox(51, 50), (51.0, 50.0))], 2, frame
    )

    assert results[2] != first_canonical
    assert c.get_unique_count() == 2




# ---------------------------------------------------------------------------
# Same-frame canonical uniqueness (regardless of call order)
# ---------------------------------------------------------------------------

def test_simultaneous_tracks_do_not_merge_order_a():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    results = c.register_frame(
        [
            (1, 2, _bbox(50, 50), (50.0, 50.0)),
            (2, 2, _bbox(200, 200), (200.0, 200.0)),
        ],
        1,
        frame,
    )
    assert results[1] != results[2]
    assert c.get_unique_count() == 2


def test_simultaneous_tracks_do_not_merge_order_b():
    """Reversed registration order must produce the same distinct IDs."""
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    results = c.register_frame(
        [
            (2, 2, _bbox(200, 200), (200.0, 200.0)),
            (1, 2, _bbox(50, 50), (50.0, 50.0)),
        ],
        1,
        frame,
    )
    assert results[1] != results[2]
    assert c.get_unique_count() == 2


def test_simultaneous_tracks_same_canonical_rejected():
    """Two new tracks that both match one canonical: first wins, second fragments."""
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(5, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    results = c.register_frame(
        [
            (7, 2, _bbox(52, 52), (52.0, 52.0)),
            (8, 2, _bbox(53, 53), (53.0, 53.0)),
        ],
        2,
        frame,
    )
    assert results[7] == c.get_canonical_id(5)
    assert results[8] != c.get_canonical_id(5)
    assert c.get_unique_count() == 2


# ---------------------------------------------------------------------------
# Class gate: class ID 0 and mismatches
# ---------------------------------------------------------------------------

def test_class_id_zero_matches_class_id_zero():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(1, 0, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    cid0 = c.get_canonical_id(1)
    results = c.register_frame([(2, 0, _bbox(51, 51), (51.0, 51.0))], 2, frame)
    assert results[2] == cid0
    assert c.get_unique_count() == 1


def test_class_mismatch_never_matches():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(1, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    results = c.register_frame([(2, 3, _bbox(51, 51), (51.0, 51.0))], 2, frame)
    assert results[2] != c.get_canonical_id(1)


# ---------------------------------------------------------------------------
# Class flips: registered class is stable
# ---------------------------------------------------------------------------

def test_class_flip_preserves_registered_class():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(1, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    cid = c.get_canonical_id(1)
    rec = c._canonicals[cid]
    registered = rec.registered_class_id
    assert registered == 2
    c.register_frame([(1, 3, _bbox(51, 51), (51.0, 51.0))], 2, frame)
    rec = c._canonicals[cid]
    assert rec.registered_class_id == registered
    assert 3 in rec.class_history


# ---------------------------------------------------------------------------
# ID switch / fragmented track rejoins
# ---------------------------------------------------------------------------

def test_id_switch_resolves_to_existing_canonical():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(5, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    cid5 = c.get_canonical_id(5)
    results = c.register_frame([(9, 2, _bbox(51, 51), (51.0, 51.0))], 2, frame)
    assert results[9] == cid5
    assert c.get_unique_count() == 1


# ---------------------------------------------------------------------------
# Missing appearance evidence (frame=None)
# ---------------------------------------------------------------------------

def test_frame_none_creates_new_canonical_no_match():
    c = CanonicalCounter(fps=25.0)
    c.register_frame([(1, 2, _bbox(50, 50), (50.0, 50.0))], 1, None)
    results = c.register_frame([(2, 2, _bbox(51, 51), (51.0, 51.0))], 2, None)
    assert results[2] != c.get_canonical_id(1)
    assert c.get_unique_count() == 2


def test_frame_none_registered_track_keeps_mapping():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(1, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    cid = c.get_canonical_id(1)
    results = c.register_frame([(1, 2, _bbox(51, 51), (51.0, 51.0))], 2, None)
    assert results[1] == cid
    assert c.get_unique_count() == 1


# ---------------------------------------------------------------------------
# Motion / gap gates
# ---------------------------------------------------------------------------

def test_large_gap_creates_new_canonical():
    c = CanonicalCounter(fps=25.0, max_gap_seconds=0.1)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(1, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    results = c.register_frame([(2, 2, _bbox(51, 51), (51.0, 51.0))], 11, frame)
    assert results[2] != c.get_canonical_id(1)
    assert c.get_unique_count() == 2


def test_implausible_speed_prevents_reidentification():
    c = CanonicalCounter(fps=25.0, max_speed_px_per_sec=10.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(1, 2, _bbox(20, 50), (20.0, 50.0))], 1, frame)

    results = c.register_frame(
        [(2, 2, _bbox(40, 50), (40.0, 50.0))],
        2,
        frame,
    )

    assert results[2] != c.get_canonical_id(1)
    assert c.get_frame_diagnostics()[0].reason == "motion_or_gap_rejected"


def test_ambiguous_candidates_create_new_canonical():
    c = CanonicalCounter(fps=25.0, max_speed_px_per_sec=5000.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame(
        [
            (1, 2, _bbox(40, 50), (40.0, 50.0)),
            (2, 2, _bbox(60, 50), (60.0, 50.0)),
        ],
        1,
        frame,
    )

    results = c.register_frame(
        [(3, 2, _bbox(50, 50), (50.0, 50.0))],
        2,
        frame,
    )

    assert results[3] not in {
        c.get_canonical_id(1),
        c.get_canonical_id(2),
    }
    decision = c.get_frame_diagnostics()[0]
    assert decision.matched_existing is False
    assert decision.reason == "ambiguous_candidates"
    assert decision.similarity == 1.0


def test_empty_frame_no_change():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(1, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    before = c.get_unique_count()
    c.register_frame([], 2, frame)
    assert c.get_unique_count() == before


# ---------------------------------------------------------------------------
# get_canonical_id
# ---------------------------------------------------------------------------

def test_get_canonical_id_unseen_returns_none():
    c = CanonicalCounter(fps=25.0)
    assert c.get_canonical_id(999) is None


# ---------------------------------------------------------------------------
# Detections without tracker IDs are not registered here
# ---------------------------------------------------------------------------

def test_no_id_detections_never_registered():
    c = CanonicalCounter(fps=25.0)
    assert c.get_unique_count() == 0
    c.register_frame([], 1, None)
    assert c.get_unique_count() == 0


def test_register_single_call_same_frame_uniqueness():
    """Multiple register() calls for the same frame respect claim tracking."""
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register(1, 2, _bbox(50, 50), (50.0, 50.0), 1, frame)
    c.register(2, 2, _bbox(200, 200), (200.0, 200.0), 1, frame)
    assert c.get_canonical_id(1) != c.get_canonical_id(2)
    assert c.get_unique_count() == 2


def test_repeated_observations_update_existing_canonical():
    c = CanonicalCounter(fps=25.0)
    frame = _frame_with_color((30, 100, 200))
    c.register_frame([(5, 2, _bbox(50, 50), (50.0, 50.0))], 1, frame)
    c.register_frame([(5, 2, _bbox(51, 51), (51.0, 51.0))], 2, frame)
    assert c.get_unique_count() == 1
    rec = c._canonicals[c.get_canonical_id(5)]
    assert len(rec.appearances) == 2
    assert rec.last_frame == 2


def _bbox(cx, cy, w=40, h=60):
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)