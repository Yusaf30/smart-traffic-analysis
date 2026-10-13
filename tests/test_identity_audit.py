"""Tests for traffic_analysis.identity_audit (synthetic logs only)."""

import contextlib
import csv
import hashlib
import io
import json
import os

import pytest

from traffic_analysis.identity_audit import (
    APPROVALS_SNAPSHOT,
    OUTPUT_FILES,
    ApprovalError,
    AuditError,
    AuditParams,
    build_identity_map,
    find_merge_candidates,
    find_overlaps,
    load_approvals,
    load_track_log,
    main,
    run_audit,
    summarize_tracks,
)
from traffic_analysis.track_log import TrackLogger


def bx(cx, cy, w=60.0, h=40.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def make_log(path, rows):
    """rows: (frame, track_id, class_id, box)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = TrackLogger(path)
    try:
        for frame, tid, cls, box in rows:
            logger.write(frame, tid, cls, 0.9, box)
    finally:
        logger.close()
    return path


def write_approvals(path, entries):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("from_id", "to_id", "decision", "reviewer", "evidence_note"))
        for entry in entries:
            writer.writerow(entry)
    return path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tracks_of(tmp_path, rows):
    return load_track_log(make_log(tmp_path / "raw" / "track_log.csv", rows))


def short_pair_rows():
    """A 51/52-like situation: two very short tracks one frame apart."""
    return [
        (196, 51, 2, bx(600, 300)),
        (197, 51, 3, bx(605, 302)),
        (199, 52, 2, bx(612, 305)),
    ]


def overlap_pair_rows():
    """A 53/54-like situation: two tracks sharing frames 200-202."""
    rows = [(f, 53, 2, bx(1100, 600)) for f in (200, 201, 202)]
    rows += [(f, 54, 2, bx(1110, 605)) for f in range(200, 208)]
    return sorted(rows)


# ---------------------------------------------------------------- loading

def test_load_rejects_wrong_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("frame,id\n1,2\n", encoding="utf-8")
    with pytest.raises(AuditError, match="header"):
        load_track_log(path)


def test_load_rejects_duplicate_observation(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(
        "frame,track_id,class_id,conf,x1,y1,x2,y2\n"
        "1,5,2,0.9,0,0,10,10\n"
        "1,5,2,0.9,1,1,11,11\n", encoding="utf-8")
    with pytest.raises(AuditError, match="duplicate"):
        load_track_log(path)


def test_load_rejects_non_finite_and_bad_columns(tmp_path):
    nan_path = tmp_path / "nan.csv"
    nan_path.write_text(
        "frame,track_id,class_id,conf,x1,y1,x2,y2\n1,5,2,0.9,nan,0,10,10\n",
        encoding="utf-8")
    with pytest.raises(AuditError, match="non-finite"):
        load_track_log(nan_path)
    short_path = tmp_path / "short.csv"
    short_path.write_text(
        "frame,track_id,class_id,conf,x1,y1,x2,y2\n1,5,2\n", encoding="utf-8")
    with pytest.raises(AuditError, match="columns"):
        load_track_log(short_path)


def test_load_groups_and_sorts_observations(tmp_path):
    tracks = tracks_of(tmp_path, [(2, 7, 2, bx(1, 1)), (1, 7, 3, bx(1, 1)),
                                  (1, 3, 2, bx(9, 9))])
    assert sorted(tracks) == [3, 7]
    assert [o.frame for o in tracks[7].obs] == [1, 2]
    assert tracks[7].class_hist == {2: 1, 3: 1}


# ---------------------------------------------------------------- summary

def test_track_summary_fields(tmp_path):
    tracks = tracks_of(tmp_path, [
        (1, 1, 2, bx(10, 300)),            # near the left edge at first frame
        (2, 1, 2, bx(100, 300)),
        (4, 1, 3, bx(200, 300)),           # frame 3 missing inside the span
        (5, 2, 2, bx(640, 360)),
    ])
    rows = {int(r["track_id"]): r for r in
            summarize_tracks(tracks, AuditParams())}
    assert rows[1]["n_obs"] == 3
    assert rows[1]["frames_missing_in_span"] == 1
    assert rows[1]["class_hist"] == "2:2|3:1"
    assert rows[1]["modal_class"] == 2
    assert rows[1]["in_frame_1"] == "true"
    assert rows[1]["first_near_edge"] == "true"
    assert rows[1]["last_near_edge"] == "false"
    assert rows[1]["short_track"] == "false"
    assert rows[2]["short_track"] == "true"
    assert rows[2]["in_frame_1"] == "false"


# ------------------------------------------------------------- candidates

def test_short_gap_candidate_flags_weak_evidence(tmp_path):
    tracks = tracks_of(tmp_path, short_pair_rows())
    rows = find_merge_candidates(tracks, AuditParams())
    assert [(r["from_id"], r["to_id"]) for r in rows] == [(51, 52)]
    row = rows[0]
    assert row["frame_gap"] == 2 and row["missing_frames"] == 1
    assert row["motion_flag"] == "insufficient"
    assert row["motion_resid_fwd_norm"] == "" and row["motion_resid_bwd_norm"] == ""
    flags = row["evidence_flags"].split(";")
    assert "short_from_track" in flags and "short_to_track" in flags
    assert "motion_insufficient" in flags
    assert row["decision"] == ""
    assert row["review_ref"] == "ids 51->52; frames 197 and 199"


def test_overlapping_tracks_are_never_candidates_but_are_listed(tmp_path):
    tracks = tracks_of(tmp_path, overlap_pair_rows())
    assert find_merge_candidates(tracks, AuditParams()) == []
    rows = find_overlaps(tracks)
    assert len(rows) == 1
    row = rows[0]
    assert (row["id_a"], row["id_b"]) == (53, 54)
    assert row["relation"] == "coexisting"
    assert row["shared_frames"] == 3
    assert (row["first_shared_frame"], row["last_shared_frame"]) == (200, 202)
    assert float(row["max_iou"]) > 0.5
    assert row["decision"] == ""


def test_interleaved_tracks_without_shared_frames_are_listed(tmp_path):
    tracks = tracks_of(tmp_path, [
        (1, 1, 2, bx(100, 100)), (3, 1, 2, bx(100, 100)),
        (2, 2, 2, bx(100, 100)), (4, 2, 2, bx(100, 100)),
    ])
    rows = find_overlaps(tracks)
    assert [r["relation"] for r in rows] == ["interleaved"]
    assert rows[0]["shared_frames"] == 0 and rows[0]["mean_iou"] == ""
    assert find_merge_candidates(tracks, AuditParams()) == []


def test_gap_beyond_limit_is_excluded(tmp_path):
    tracks = tracks_of(tmp_path, [
        (1, 1, 2, bx(100, 100)), (3, 1, 2, bx(100, 100)),
        (40, 2, 2, bx(100, 100)),
    ])
    assert find_merge_candidates(tracks, AuditParams()) == []
    rows = find_merge_candidates(tracks, AuditParams(max_frame_gap=40))
    assert [(r["from_id"], r["to_id"]) for r in rows] == [(1, 2)]


def test_far_displacement_is_excluded_unless_limit_raised(tmp_path):
    tracks = tracks_of(tmp_path, [
        (1, 1, 2, bx(100, 100)), (5, 2, 2, bx(1100, 100)),
    ])
    assert find_merge_candidates(tracks, AuditParams()) == []
    rows = find_merge_candidates(tracks, AuditParams(max_norm_displacement=1000))
    assert len(rows) == 1


def test_ambiguous_successors_are_flagged(tmp_path):
    rows_in = [(f, 1, 2, bx(300, 300)) for f in range(1, 6)]
    rows_in += [(f, 2, 2, bx(320, 300)) for f in range(8, 13)]
    rows_in += [(f, 3, 2, bx(330, 310)) for f in range(9, 13)]
    tracks = tracks_of(tmp_path, sorted(rows_in))
    rows = find_merge_candidates(tracks, AuditParams())
    pairs = [(r["from_id"], r["to_id"]) for r in rows]
    assert pairs == [(1, 2), (1, 3)]
    for row in rows:
        assert row["successor_candidates"] == 2
        assert "ambiguous_successor" in row["evidence_flags"].split(";")


def test_ambiguous_predecessors_are_flagged(tmp_path):
    rows_in = [(f, 1, 2, bx(300, 300)) for f in range(1, 6)]
    rows_in += [(f, 2, 2, bx(305, 300)) for f in range(2, 6)]
    rows_in += [(f, 3, 2, bx(320, 300)) for f in range(9, 14)]
    tracks = tracks_of(tmp_path, sorted(rows_in))
    rows = find_merge_candidates(tracks, AuditParams())
    pairs = [(r["from_id"], r["to_id"]) for r in rows]
    assert pairs == [(1, 3), (2, 3)]
    for row in rows:
        assert row["predecessor_candidates"] == 2
        assert "ambiguous_predecessor" in row["evidence_flags"].split(";")


def test_class_difference_is_listed_as_evidence_not_dropped(tmp_path):
    tracks = tracks_of(tmp_path, [
        (1, 1, 2, bx(100, 100)), (4, 2, 3, bx(105, 100)),
    ])
    rows = find_merge_candidates(tracks, AuditParams())
    assert len(rows) == 1
    assert rows[0]["class_compat"] == "different"
    assert "class_different" in rows[0]["evidence_flags"].split(";")


def test_motion_consistency_uses_velocity_when_enough_observations(tmp_path):
    moving = [(f, 1, 2, bx(100 + 10 * (f - 1), 300, 50, 50)) for f in range(1, 6)]
    # Track 1 ends at x=140 in frame 5; at 10 px/frame it reaches x=160 by frame 7.
    on_path = tracks_of(tmp_path, moving + [(7, 2, 2, bx(160, 300, 50, 50))])
    row = find_merge_candidates(on_path, AuditParams())[0]
    assert row["motion_flag"] == "consistent"
    assert float(row["motion_resid_fwd_norm"]) == pytest.approx(0.0, abs=1e-6)
    assert row["motion_resid_bwd_norm"] == ""


def test_motion_inconsistency_is_flagged(tmp_path):
    moving = [(f, 1, 2, bx(100 + 10 * (f - 1), 300, 50, 50)) for f in range(1, 6)]
    off_path = make_log(
        tmp_path / "off" / "track_log.csv",
        moving + [(7, 2, 2, bx(360, 300, 50, 50))])
    row = find_merge_candidates(load_track_log(off_path), AuditParams())[0]
    assert row["motion_flag"] == "inconsistent"
    assert float(row["motion_resid_fwd_norm"]) > 1.0
    assert "motion_inconsistent" in row["evidence_flags"].split(";")


def test_edge_adjacency_flags(tmp_path):
    tracks = tracks_of(tmp_path, [
        (1, 1, 2, bx(30, 300)),            # x1 = 0, touches the left border
        (3, 2, 2, bx(60, 300)),            # not near any border
    ])
    row = find_merge_candidates(tracks, AuditParams())[0]
    assert row["a_end_near_edge"] == "true"
    assert row["b_start_near_edge"] == "false"
    assert "edge_adjacent_end" in row["evidence_flags"].split(";")


# -------------------------------------------------------------- approvals

def test_approvals_rejects_invalid_files(tmp_path):
    tracks = tracks_of(tmp_path, short_pair_rows() + overlap_pair_rows())
    cases = {
        "unknown id": [(51, 999, "approved", "u", "n")],
        "bad decision": [(51, 52, "maybe", "u", "n")],
        "same id": [(51, 51, "approved", "u", "n")],
        "non integer": [("x", 52, "approved", "u", "n")],
        "duplicate pair": [(51, 52, "approved", "u", "n"),
                           (52, 51, "unresolved", "u", "n")],
        "coexisting approved": [(53, 54, "approved", "u", "n")],
        "wrong order approved": [(52, 51, "approved", "u", "n")],
    }
    for name, entries in cases.items():
        path = write_approvals(tmp_path / f"{name.replace(' ', '_')}.csv", entries)
        with pytest.raises(ApprovalError):
            load_approvals(path, tracks)


def test_approvals_rejects_branching_and_bad_header(tmp_path):
    rows = [(1, 1, 2, bx(100, 100)), (5, 2, 2, bx(100, 100)),
            (6, 3, 2, bx(100, 100))]
    tracks = tracks_of(tmp_path, rows)
    branch = write_approvals(tmp_path / "branch.csv", [
        (1, 2, "approved", "u", "n"), (1, 3, "approved", "u", "n")])
    with pytest.raises(ApprovalError, match="successor"):
        load_approvals(branch, tracks)
    merge = write_approvals(tmp_path / "merge.csv", [
        (1, 3, "approved", "u", "n"), (2, 3, "approved", "u", "n")])
    with pytest.raises(ApprovalError, match="predecessor"):
        load_approvals(merge, tracks)
    header = tmp_path / "header.csv"
    header.write_text("from_id,to_id\n1,2\n", encoding="utf-8")
    with pytest.raises(ApprovalError, match="header"):
        load_approvals(header, tracks)


def test_unresolved_and_rejected_do_not_change_mapping(tmp_path):
    tracks = tracks_of(tmp_path, short_pair_rows() + overlap_pair_rows())
    path = write_approvals(tmp_path / "a.csv", [
        (53, 54, "unresolved", "u", "overlap"), (51, 52, "rejected", "u", "no")])
    approvals = load_approvals(path, tracks)
    mapping = build_identity_map(tracks, approvals)
    assert mapping == {51: 51, 52: 52, 53: 53, 54: 54}


def test_approved_chain_maps_to_earliest_track(tmp_path):
    rows = [(1, 1, 2, bx(100, 100)), (5, 2, 2, bx(100, 100)),
            (9, 3, 3, bx(100, 100)), (20, 4, 2, bx(900, 100))]
    tracks = tracks_of(tmp_path, rows)
    path = write_approvals(tmp_path / "a.csv", [
        (1, 2, "approved", "u", "n"), (2, 3, "approved", "u", "n")])
    mapping = build_identity_map(tracks, load_approvals(path, tracks))
    assert mapping == {1: 1, 2: 1, 3: 1, 4: 4}


# ------------------------------------------------------------- end to end

def seeded_setup(tmp_path):
    raw = make_log(tmp_path / "raw" / "track_log.csv",
                   short_pair_rows() + overlap_pair_rows())
    approvals = write_approvals(tmp_path / "approved_mappings.csv", [
        (51, 52, "approved", "user", "visual review"),
        (53, 54, "unresolved", "user", "overlap; do not merge automatically"),
    ])
    return raw, approvals


def test_run_audit_applies_only_approved_mapping_and_keeps_raw_intact(tmp_path):
    raw, approvals = seeded_setup(tmp_path)
    raw.chmod(0o444)
    before = (raw.read_bytes(), raw.stat().st_mtime_ns,
              sorted(p.name for p in raw.parent.iterdir()))

    report = run_audit(raw, tmp_path / "analysis", approvals)

    after = (raw.read_bytes(), raw.stat().st_mtime_ns,
             sorted(p.name for p in raw.parent.iterdir()))
    assert before == after
    out = tmp_path / "analysis"
    assert sorted(p.name for p in out.iterdir()) == sorted(
        OUTPUT_FILES + (APPROVALS_SNAPSHOT,))
    assert report["raw_log"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()

    id_map = {int(r["track_id"]): r for r in read_csv(out / "identity_map.csv")}
    assert id_map[51]["canonical_id"] == "51" and id_map[52]["canonical_id"] == "51"
    assert id_map[52]["source"] == "approved_mapping"
    assert id_map[53]["canonical_id"] == "53" and id_map[54]["canonical_id"] == "54"
    assert id_map[53]["source"] == "identity"

    groups = {int(r["canonical_id"]): r for r in read_csv(out / "identity_groups.csv")}
    assert groups[51]["member_ids"] == "51|52"
    assert groups[51]["n_obs"] == "3"
    assert groups[51]["class_hist"] == "2:2|3:1"
    assert groups[51]["class_conflict"] == "true"

    merges = read_csv(out / "merge_proposals.csv")
    assert [(r["from_id"], r["to_id"], r["decision"]) for r in merges] == [
        ("51", "52", "approved")]
    overlaps = read_csv(out / "overlap_pairs.csv")
    assert [(r["id_a"], r["id_b"], r["decision"]) for r in overlaps] == [
        ("53", "54", "unresolved")]

    saved = json.loads((out / "audit_report.json").read_text(encoding="utf-8"))
    assert saved["approvals"]["approved_not_in_candidates"] == []
    assert saved["counts"]["mapped_tracks"] == 2
    assert saved["counts"]["coexisting_pairs"] == 1
    assert (out / APPROVALS_SNAPSHOT).read_bytes() == approvals.read_bytes()


def test_run_audit_is_deterministic(tmp_path):
    raw, approvals = seeded_setup(tmp_path)
    run_audit(raw, tmp_path / "one", approvals)
    run_audit(raw, tmp_path / "two", approvals)
    for name in OUTPUT_FILES + (APPROVALS_SNAPSHOT,):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_run_audit_without_approvals_maps_nothing(tmp_path):
    raw, _ = seeded_setup(tmp_path)
    report = run_audit(raw, tmp_path / "analysis")
    assert report["approvals"] is None
    assert report["counts"]["mapped_tracks"] == 0
    assert not (tmp_path / "analysis" / APPROVALS_SNAPSHOT).exists()
    assert report["counts"]["canonical_identities"] == report["counts"]["tracks"]


def test_sidecar_mismatch_is_refused_and_leaves_no_output(tmp_path):
    raw, _ = seeded_setup(tmp_path)
    (raw.parent / "track_log.csv.sha256").write_text(
        "0" * 64 + "  track_log.csv\n", encoding="utf-8")
    with pytest.raises(AuditError, match="does not match"):
        run_audit(raw, tmp_path / "analysis")
    assert not (tmp_path / "analysis").exists()


def test_sidecar_match_is_reported_as_verified(tmp_path):
    raw, _ = seeded_setup(tmp_path)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    (raw.parent / "track_log.csv.sha256").write_text(
        f"{digest}  track_log.csv\n", encoding="utf-8")
    report = run_audit(raw, tmp_path / "analysis")
    assert report["raw_log"]["status"] == "verified"


def test_expect_sha256_mismatch_is_refused(tmp_path):
    raw, _ = seeded_setup(tmp_path)
    with pytest.raises(AuditError, match="expect-sha256"):
        run_audit(raw, tmp_path / "analysis", expect_sha256="f" * 64)
    assert not (tmp_path / "analysis").exists()


def test_output_inside_raw_directory_is_refused(tmp_path):
    raw, _ = seeded_setup(tmp_path)
    with pytest.raises(AuditError, match="raw log directory"):
        run_audit(raw, raw.parent / "analysis")
    with pytest.raises(AuditError, match="raw log directory"):
        run_audit(raw, raw.parent)


def test_existing_output_directory_is_refused(tmp_path):
    raw, _ = seeded_setup(tmp_path)
    (tmp_path / "analysis").mkdir()
    with pytest.raises(AuditError, match="already exists"):
        run_audit(raw, tmp_path / "analysis")


def test_header_only_log_is_refused(tmp_path):
    raw = make_log(tmp_path / "raw" / "track_log.csv", [])
    with pytest.raises(AuditError, match="no observations"):
        run_audit(raw, tmp_path / "analysis")


def test_invalid_approvals_leave_no_output(tmp_path):
    raw, _ = seeded_setup(tmp_path)
    bad = write_approvals(tmp_path / "bad.csv", [(51, 999, "approved", "u", "n")])
    with pytest.raises(ApprovalError):
        run_audit(raw, tmp_path / "analysis", bad)
    assert not (tmp_path / "analysis").exists()


def test_params_validation():
    for kwargs in ({"max_frame_gap": 0}, {"velocity_window": 1},
                   {"edge_margin": -1.0}, {"motion_tolerance": -0.1}):
        with pytest.raises(ValueError):
            AuditParams(**kwargs)


def test_main_success_and_failure_codes(tmp_path):
    raw, approvals = seeded_setup(tmp_path)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--raw-log", str(raw), "--approvals", str(approvals),
                     "--out-root", str(tmp_path / "root")])
    assert code == 0
    created = [p.name for p in (tmp_path / "root").iterdir()]
    assert len(created) == 1 and created[0].startswith("analysis-")
    assert "merge_candidates: 1" in out.getvalue()

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = main(["--raw-log", str(raw), "--out-dir", str(tmp_path / "x"),
                     "--expect-sha256", "f" * 64])
    assert code == 2
    assert "audit failed" in err.getvalue()
    assert not os.path.exists(tmp_path / "x")

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = main(["--raw-log", str(raw), "--out-dir", str(tmp_path / "y"),
                     "--max-frame-gap", "0"])
    assert code == 2
    assert "invalid parameter" in err.getvalue()
