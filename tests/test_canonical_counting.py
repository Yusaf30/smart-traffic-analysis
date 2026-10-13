import csv
import os

import pytest

from traffic_analysis.canonical_counting import (
    CanonicalMapError,
    LineCounter,
    load_canonical_map,
)

NAMES = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
LINE = 330
HDR = "track_id,canonical_id,group_size,source\n"


def write_map(tmp_path, body, header=HDR):
    p = tmp_path / "identity_map.csv"
    p.write_text(header + body)
    return p


def run(counter, obs):
    return [e for o in obs if (e := counter.update(*o))]


# Mirrors the real map: ordinary "identity" rows plus approved merge rows.
FULL = (
    "7,7,1,identity\n"
    "51,51,2,approved_mapping\n"
    "52,51,2,approved_mapping\n"
    "53,53,1,identity\n"
    "54,54,1,identity\n"
)


def test_full_map_with_identity_and_approved_rows(tmp_path):
    m = load_canonical_map(write_map(tmp_path, FULL))
    assert m == {7: 7, 51: 51, 52: 51, 53: 53, 54: 54}


def test_51_52_resolve_to_51(tmp_path):
    m = load_canonical_map(write_map(tmp_path, "51,51,2,approved_mapping\n52,51,2,approved_mapping\n"))
    assert m == {51: 51, 52: 51}


def test_unmapped_and_unresolved_stay_separate(tmp_path):
    c = LineCounter(LINE, NAMES, load_canonical_map(write_map(tmp_path, FULL)))
    assert c.canonical.get(53, 53) == 53 and c.canonical.get(54, 54) == 54
    assert c.canonical.get(999, 999) == 999  # id absent from the map


def test_53_54_counted_separately_with_full_map(tmp_path):
    c = LineCounter(LINE, NAMES, load_canonical_map(write_map(tmp_path, FULL)))
    run(c, [(1, 53, 2, 0, 320), (1, 54, 2, 50, 320), (2, 53, 2, 0, 340), (2, 54, 2, 50, 340)])
    assert c.counts["Car"] == 2 and c.conflicts == []


def test_duplicate_identical_rows_are_accepted(tmp_path):
    m = load_canonical_map(write_map(tmp_path, "7,7,1,identity\n7,7,1,identity\n"))
    assert m == {7: 7}


@pytest.mark.parametrize("body,header", [
    ("x,y,2,approved_mapping\n", HDR),                                # non-integer ids
    ("52,51,2,auto_proposal\n", HDR),                                 # unknown source
    ("52,51,2,\n", HDR),                                              # empty source
    ("52,51,1,identity\n", HDR),                                      # identity with tid != cid
    ("52,51,2,approved_mapping\n52,50,2,approved_mapping\n", HDR),     # conflicting approved rows
    ("52,52,1,identity\n52,51,2,approved_mapping\n", HDR),             # identity vs approved conflict
    ("52,51,2,approved_mapping\n51,50,2,approved_mapping\n", HDR),     # chain
    ("52,51,2,approved_mapping\n51,51,1,identity\n53,52,2,approved_mapping\n", HDR),  # chain via 52
    ("52,51\n", "track_id,canonical_id\n"),                          # missing columns
    ("", HDR),                                                       # empty map
])
def test_bad_map_raises(tmp_path, body, header):
    with pytest.raises(CanonicalMapError):
        load_canonical_map(write_map(tmp_path, body, header))


def test_empty_map_allowed_only_when_not_required(tmp_path):
    p = write_map(tmp_path, "")
    assert load_canonical_map(p, require_nonempty=False) == {}


def test_missing_map_raises(tmp_path):
    with pytest.raises(CanonicalMapError):
        load_canonical_map(tmp_path / "nope.csv")


# Fragment A crosses (counted), jitters back, new id B crosses again.
JITTER = [(1, 51, 2, 10, 320), (2, 51, 2, 10, 335), (3, 51, 2, 10, 328),
          (5, 52, 2, 10, 329), (6, 52, 2, 10, 336)]
# Crossing inside a long lost-frames gap is kept unconfirmed for audit.
GAP = [(1, 51, 2, 10, 320), (2, 51, 2, 10, 326), (5, 52, 2, 10, 336)]


def test_jitter_double_count_raw_vs_canonical():
    assert len(run(LineCounter(LINE, NAMES), JITTER)) == 2
    c = LineCounter(LINE, NAMES, {51: 51, 52: 51})
    ev = run(c, JITTER)
    assert len(ev) == 1 and c.counts["Car"] == 1
    assert ev[0]["raw_id"] == 51 and ev[0]["canonical_id"] == 51


def test_gap_crossing_is_unconfirmed_for_raw_and_canonical():
    assert run(LineCounter(LINE, NAMES), GAP) == []
    c = LineCounter(LINE, NAMES, {51: 51, 52: 51})
    assert run(c, GAP) == []
    assert c.counts["Car"] == 0
    assert c.unconfirmed_crossings == [{
        "frame": 5, "raw_id": 52, "canonical_id": 51,
        "reason": "detection_gap",
    }]


def test_box_jump_crossing_is_unconfirmed_and_audited():
    c = LineCounter(LINE, NAMES)
    c.update(1, 9, 2, 100, 320, box=(50, 270, 150, 370))
    event = c.update(2, 9, 2, 100, 340, box=(0, 240, 200, 440))

    assert event is None and c.counts["Car"] == 0
    assert c.unconfirmed_crossings[0]["reason"] == "box_size_jump"


def test_gap_candidate_does_not_duplicate_a_later_valid_crossing():
    c = LineCounter(LINE, NAMES)
    run(c, [(1, 9, 2, 0, 320), (2, 9, 2, 0, 326), (5, 9, 2, 0, 336),
            (6, 9, 2, 0, 340), (7, 9, 2, 0, 320), (8, 9, 2, 0, 340)])

    assert c.counts["Car"] == 1
    assert len(c.unconfirmed_crossings) == 1


def test_repeated_crossings_count_once():
    c = LineCounter(LINE, NAMES)
    run(c, [(1, 9, 2, 0, 320), (2, 9, 2, 0, 340), (3, 9, 2, 0, 320), (4, 9, 2, 0, 340)])
    assert c.counts["Car"] == 1


def test_same_frame_conflict_is_flagged_not_merged_silently():
    c = LineCounter(LINE, NAMES, {51: 51, 52: 51})
    run(c, [(1, 51, 2, 0, 320), (1, 52, 2, 0, 340)])
    assert c.conflicts == [(1, 51, 51, 52)] and c.counts["Car"] == 0


def test_disabled_equals_identity_and_event_format():
    c = LineCounter(LINE, NAMES)
    ev = run(c, [(1, 5, 3, 0, 320), (2, 5, 3, 0, 340)])
    assert set(c.counts) == {"Car", "Bike", "Bus", "Truck"} and c.counts["Bike"] == 1
    assert ev[0] == {"frame": 2, "raw_id": 5, "canonical_id": 5, "vehicle_class": "Bike"}


# --- Vertical (X) counting line: cumulative totals, once-only, audit logging ---

def test_vertical_line_counts_cumulative_and_once_per_track():
    # Horizontal traffic across a vertical line at x=640 (india_trafic-style).
    c = LineCounter(640, NAMES, axis="x")
    ev = run(c, [
        (1, 10, 3, 600, 450), (2, 10, 3, 660, 450),   # bike crosses L->R
        (3, 11, 2, 660, 450), (4, 11, 2, 600, 450),   # car crosses R->L
        (5, 10, 3, 700, 450), (6, 10, 3, 760, 450),   # tid 10 must NOT count again
    ])
    assert len(ev) == 2
    assert c.counts == {"Car": 1, "Bike": 1, "Bus": 0, "Truck": 0}
    assert c.counts["Car"] + c.counts["Bike"] + c.counts["Bus"] + c.counts["Truck"] == 2
    assert c.unconfirmed_crossings == []


def test_vertical_line_staying_on_one_side_counts_nothing():
    c = LineCounter(640, NAMES, axis="x")
    ev = run(c, [
        (1, 20, 2, 300, 450), (2, 20, 2, 400, 450), (3, 20, 2, 500, 450),
    ])
    assert ev == [] and sum(c.counts.values()) == 0


def test_vertical_line_implausible_jump_is_audited_not_counted():
    c = LineCounter(640, NAMES, axis="x")
    # Boxes present so the center-jump check runs: a 300px teleport across the
    # line in one frame is far beyond 0.25 * box-diagonal, so it is audited.
    ev = run(c, [
        (1, 30, 2, 600, 450, (570, 400, 630, 500)),
        (2, 30, 2, 900, 450, (870, 400, 930, 500)),
    ])
    assert ev == []
    assert sum(c.counts.values()) == 0
    assert c.unconfirmed_crossings == [{
        "frame": 2, "raw_id": 30, "canonical_id": 30, "reason": "center_jump",
    }]


def test_vertical_line_box_jump_is_audited_not_counted():
    c = LineCounter(640, NAMES, axis="x")
    c.update(1, 31, 2, 620, 450, box=(570, 400, 670, 500))
    ev = c.update(2, 31, 2, 660, 450, box=(510, 300, 810, 600))  # box_size_jump
    assert ev is None and sum(c.counts.values()) == 0
    assert c.unconfirmed_crossings[0]["reason"] == "box_size_jump"


def test_vertical_line_detection_gap_is_audited_not_counted():
    c = LineCounter(640, NAMES, axis="x")
    ev = run(c, [(1, 32, 2, 600, 450), (6, 32, 2, 680, 450)])  # gap > 2 frames
    assert ev == [] and sum(c.counts.values()) == 0
    assert c.unconfirmed_crossings[0]["reason"] == "detection_gap"


RAW = os.environ.get("TRAFFIC_RAW_LOG")
MAP = os.environ.get("TRAFFIC_ID_MAP")


@pytest.mark.skipif(not (RAW and MAP), reason="set TRAFFIC_RAW_LOG and TRAFFIC_ID_MAP")
def test_frozen_log_replay_matches_baseline():
    with open(RAW, newline="") as f:
        rows = [(int(r["frame"]), int(r["track_id"]), int(r["class_id"]),
                 (int(float(r["x1"])) + int(float(r["x2"]))) // 2,
                 (int(float(r["y1"])) + int(float(r["y2"]))) // 2,
                 tuple(int(float(r[k])) for k in ("x1", "y1", "x2", "y2")))
                for r in csv.DictReader(f)]
    base = LineCounter(LINE, NAMES)
    fixed = LineCounter(LINE, NAMES, load_canonical_map(MAP))
    run(base, rows)
    run(fixed, rows)
    expected = {"Car": 9, "Bike": 0, "Bus": 0, "Truck": 0}
    assert base.counts == expected and fixed.counts == expected
    assert fixed.conflicts == []
    unconfirmed_ids = {row["raw_id"] for row in fixed.unconfirmed_crossings}
    assert {47, 61, 67, 79, 84} <= unconfirmed_ids
    assert all(row["reason"] == "detection_gap" for row in fixed.unconfirmed_crossings)