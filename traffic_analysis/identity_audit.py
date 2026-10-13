"""Identity audit for the raw Pass 1 tracker log.

Reads a frozen ``track_log.csv`` (never modifies it), reports per-track
evidence, coexisting tracks and *proposed* fragment links, and applies only
human-approved mappings, in a separate analysis layer.

Nothing here merges identities automatically. Every candidate link is a
proposal with evidence for human review. A mapping takes effect only when it
appears in an approvals file with ``decision == approved``.

Usage (from the project root):

    python -m traffic_analysis.identity_audit \\
        --raw-log ~/traffic-pass1/raw-<ts>/out/track_log.csv \\
        --approvals ~/traffic-pass1/approved_mappings.csv \\
        --out-root ~/traffic-pass1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from traffic_analysis.track_log import TrackLogger

SCHEMA_VERSION = 1
DECISIONS = ("approved", "rejected", "unresolved")
APPROVAL_COLUMNS = ("from_id", "to_id", "decision", "reviewer", "evidence_note")

SUMMARY_COLUMNS = (
    "track_id", "first_frame", "last_frame", "n_obs", "frames_missing_in_span",
    "modal_class", "class_hist", "mean_conf", "first_box", "last_box",
    "first_near_edge", "last_near_edge", "short_track", "in_frame_1",
)
OVERLAP_COLUMNS = (
    "id_a", "id_b", "relation", "shared_frames", "first_shared_frame",
    "last_shared_frame", "mean_iou", "max_iou", "a_span", "b_span",
    "a_class_hist", "b_class_hist", "class_compat", "decision",
)
MERGE_COLUMNS = (
    "from_id", "to_id", "a_last_frame", "b_first_frame", "frame_gap",
    "missing_frames", "a_n_obs", "b_n_obs", "a_class_hist", "b_class_hist",
    "class_compat", "center_dist_px", "norm_displacement", "iou_last_first",
    "size_ratio", "motion_resid_fwd_norm", "motion_resid_bwd_norm",
    "motion_flag", "a_end_near_edge", "b_start_near_edge",
    "successor_candidates", "predecessor_candidates", "evidence_flags",
    "review_ref", "decision",
)
MAP_COLUMNS = ("track_id", "canonical_id", "group_size", "source")
GROUP_COLUMNS = (
    "canonical_id", "member_ids", "n_members", "first_frame", "last_frame",
    "n_obs", "class_hist", "modal_class", "class_conflict", "source",
)

OUTPUT_FILES = (
    "track_summary.csv",
    "overlap_pairs.csv",
    "merge_proposals.csv",
    "identity_map.csv",
    "identity_groups.csv",
    "audit_report.json",
)
APPROVALS_SNAPSHOT = "approved_mappings.used.csv"


class AuditError(ValueError):
    """The raw log, an approvals file or the output location is unusable."""


class ApprovalError(AuditError):
    """The approvals file is invalid."""


@dataclass(frozen=True)
class AuditParams:
    """Thresholds for candidate generation.

    Defaults are deliberately loose: candidates go to human review, so the
    goal is to avoid hiding plausible links, not to decide them. They were not
    tuned on any particular track.
    """

    frame_width: int = 1280
    frame_height: int = 720
    edge_margin: float = 20.0          # px from a frame border
    max_frame_gap: int = 30            # B.first_frame - A.last_frame
    max_norm_displacement: float = 4.0  # center distance / larger box diagonal
    min_obs_for_motion: int = 3        # fewer observations: no velocity estimate
    velocity_window: int = 5           # observations used for velocity
    motion_tolerance: float = 1.0      # residual / box diagonal

    def __post_init__(self) -> None:
        for name in ("frame_width", "frame_height", "max_frame_gap",
                     "min_obs_for_motion", "velocity_window"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("edge_margin", "max_norm_displacement", "motion_tolerance"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.velocity_window < 2:
            raise ValueError("velocity_window must be >= 2")


@dataclass(frozen=True)
class Obs:
    frame: int
    class_id: int
    conf: float
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def diag(self) -> float:
        x1, y1, x2, y2 = self.box
        return math.hypot(x2 - x1, y2 - y1)


@dataclass(frozen=True)
class Track:
    track_id: int
    obs: tuple[Obs, ...]  # sorted by frame, one observation per frame
    by_frame: Mapping[int, Obs] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_frame", {o.frame: o for o in self.obs})

    @property
    def first_frame(self) -> int:
        return self.obs[0].frame

    @property
    def last_frame(self) -> int:
        return self.obs[-1].frame

    @property
    def n_obs(self) -> int:
        return len(self.obs)

    @property
    def class_hist(self) -> dict[int, int]:
        return dict(sorted(Counter(o.class_id for o in self.obs).items()))

    @property
    def modal_class(self) -> int:
        return _modal(self.class_hist)

    @property
    def mean_conf(self) -> float:
        return sum(o.conf for o in self.obs) / len(self.obs)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _modal(hist: Mapping[int, int]) -> int:
    """Most frequent class; ties go to the lowest class id."""
    return sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _hist_str(hist: Mapping[int, int]) -> str:
    return "|".join(f"{c}:{n}" for c, n in sorted(hist.items()))


def _fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _box_str(box: Sequence[float]) -> str:
    return ";".join(f"{v:.1f}" for v in box)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def near_edge(box: Sequence[float], params: AuditParams) -> bool:
    x1, y1, x2, y2 = box
    m = params.edge_margin
    return (x1 <= m or y1 <= m
            or x2 >= params.frame_width - m or y2 >= params.frame_height - m)


def class_compat(a_hist: Mapping[int, int], b_hist: Mapping[int, int]) -> str:
    """Evidence only: same modal class, partial overlap, or different."""
    if _modal(a_hist) == _modal(b_hist):
        return "same"
    if set(a_hist) & set(b_hist):
        return "partial"
    return "different"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_track_log(path: str | Path) -> dict[int, Track]:
    """Read the raw log read-only; validate it; group observations by track."""
    grouped: dict[int, list[Obs]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise AuditError("raw log is empty (no header)") from None
        if tuple(header) != TrackLogger.COLUMNS:
            raise AuditError(f"unexpected raw log header: {header!r}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(TrackLogger.COLUMNS):
                raise AuditError(
                    f"line {line_no}: expected {len(TrackLogger.COLUMNS)} "
                    f"columns, got {len(row)}")
            try:
                frame, tid, cls = int(row[0]), int(row[1]), int(row[2])
                conf = float(row[3])
                box = tuple(float(v) for v in row[4:8])
            except ValueError as exc:
                raise AuditError(f"line {line_no}: {exc}") from exc
            if frame < 1:
                raise AuditError(f"line {line_no}: frame must be >= 1")
            if not all(math.isfinite(v) for v in (conf, *box)):
                raise AuditError(f"line {line_no}: non-finite value")
            if (tid, frame) in seen:
                raise AuditError(
                    f"line {line_no}: duplicate observation for track "
                    f"{tid} in frame {frame}")
            seen.add((tid, frame))
            grouped[tid].append(Obs(frame, cls, conf, box))  # type: ignore[arg-type]
    return {
        tid: Track(tid, tuple(sorted(obs, key=lambda o: o.frame)))
        for tid, obs in sorted(grouped.items())
    }


def verify_sidecar(raw: Path, actual: str, expect: str | None) -> dict:
    """Compare the raw log hash with an explicit value and/or ``<log>.sha256``."""
    status = "absent"
    sidecar = raw.with_name(raw.name + ".sha256")
    if sidecar.exists():
        tokens = sidecar.read_text(encoding="utf-8").split()
        recorded = tokens[0].lower() if tokens else ""
        if not re.fullmatch(r"[0-9a-f]{64}", recorded):
            raise AuditError(f"unreadable sidecar hash in {sidecar}")
        if recorded != actual:
            raise AuditError(
                f"raw log SHA-256 does not match {sidecar.name}: "
                f"expected {recorded}, got {actual}")
        status = "verified"
    if expect is not None:
        if expect.lower() != actual:
            raise AuditError(
                f"raw log SHA-256 does not match --expect-sha256: "
                f"expected {expect.lower()}, got {actual}")
        status = "verified" if status == "absent" else status
    return {"sidecar": sidecar.name, "status": status}


# --------------------------------------------------------------------------
# approvals
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Approval:
    from_id: int
    to_id: int
    decision: str
    reviewer: str
    evidence_note: str
    line: int


def load_approvals(path: str | Path, tracks: Mapping[int, Track]) -> list[Approval]:
    """Read and validate a human-authored approvals file."""
    approvals: list[Approval] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != set(APPROVAL_COLUMNS):
            raise ApprovalError(
                f"approvals header must contain exactly {APPROVAL_COLUMNS}, "
                f"got {tuple(reader.fieldnames or ())}")
        for line_no, row in enumerate(reader, start=2):
            if None in row or None in row.values():
                raise ApprovalError(f"line {line_no}: wrong number of columns")
            try:
                from_id, to_id = int(row["from_id"]), int(row["to_id"])
            except ValueError:
                raise ApprovalError(
                    f"line {line_no}: from_id and to_id must be integers") from None
            decision = row["decision"].strip().lower()
            if decision not in DECISIONS:
                raise ApprovalError(
                    f"line {line_no}: decision must be one of {DECISIONS}, "
                    f"got {row['decision']!r}")
            approvals.append(Approval(
                from_id, to_id, decision, row["reviewer"].strip(),
                row["evidence_note"].strip(), line_no))

    seen_pairs: dict[frozenset, int] = {}
    successors: dict[int, int] = {}
    predecessors: dict[int, int] = {}
    for a in approvals:
        for tid in (a.from_id, a.to_id):
            if tid not in tracks:
                raise ApprovalError(f"line {a.line}: track {tid} is not in the raw log")
        if a.from_id == a.to_id:
            raise ApprovalError(f"line {a.line}: from_id and to_id must differ")
        key = frozenset((a.from_id, a.to_id))
        if key in seen_pairs:
            raise ApprovalError(
                f"line {a.line}: pair {a.from_id}/{a.to_id} already listed on "
                f"line {seen_pairs[key]}")
        seen_pairs[key] = a.line
        if a.decision != "approved":
            continue
        src, dst = tracks[a.from_id], tracks[a.to_id]
        if not src.last_frame < dst.first_frame:
            raise ApprovalError(
                f"line {a.line}: approved link {a.from_id}->{a.to_id} requires "
                f"track {a.from_id} to end (frame {src.last_frame}) before "
                f"track {a.to_id} starts (frame {dst.first_frame})")
        if a.from_id in successors:
            raise ApprovalError(
                f"line {a.line}: track {a.from_id} already has an approved "
                f"successor ({successors[a.from_id]})")
        if a.to_id in predecessors:
            raise ApprovalError(
                f"line {a.line}: track {a.to_id} already has an approved "
                f"predecessor ({predecessors[a.to_id]})")
        successors[a.from_id] = a.to_id
        predecessors[a.to_id] = a.from_id
    return approvals


def build_identity_map(
    tracks: Mapping[int, Track], approvals: Sequence[Approval]
) -> dict[int, int]:
    """Map every track id to a canonical id using approved links only.

    The canonical id is the id of the earliest track in each approved chain.
    Unresolved and rejected entries never change the mapping.
    """
    succ = {a.from_id: a.to_id for a in approvals if a.decision == "approved"}
    has_pred = set(succ.values())
    canonical: dict[int, int] = {}
    for tid in sorted(tracks):
        if tid in has_pred:
            continue
        cur = tid
        canonical[cur] = tid
        while cur in succ:
            cur = succ[cur]
            canonical[cur] = tid
    if set(canonical) != set(tracks):  # pragma: no cover - guarded by load_approvals
        raise AuditError("approved links do not form simple chains")
    return canonical


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

def summarize_tracks(tracks: Mapping[int, Track], params: AuditParams) -> list[dict]:
    rows = []
    for tid in sorted(tracks):
        t = tracks[tid]
        first, last = t.obs[0], t.obs[-1]
        rows.append({
            "track_id": tid,
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "n_obs": t.n_obs,
            "frames_missing_in_span": (t.last_frame - t.first_frame + 1) - t.n_obs,
            "modal_class": t.modal_class,
            "class_hist": _hist_str(t.class_hist),
            "mean_conf": _fmt(t.mean_conf, 4),
            "first_box": _box_str(first.box),
            "last_box": _box_str(last.box),
            "first_near_edge": _bool(near_edge(first.box, params)),
            "last_near_edge": _bool(near_edge(last.box, params)),
            "short_track": _bool(t.n_obs < params.min_obs_for_motion),
            "in_frame_1": _bool(t.first_frame == 1),
        })
    return rows


def find_overlaps(
    tracks: Mapping[int, Track], decisions: Mapping[frozenset, str] | None = None
) -> list[dict]:
    """List track pairs whose frame spans overlap. Never proposals to merge."""
    decisions = decisions or {}
    ids = sorted(tracks)
    rows = []
    for i, a_id in enumerate(ids):
        a = tracks[a_id]
        for b_id in ids[i + 1:]:
            b = tracks[b_id]
            if a.first_frame > b.last_frame or b.first_frame > a.last_frame:
                continue
            shared = sorted(set(a.by_frame) & set(b.by_frame))
            ious = [iou(a.by_frame[f].box, b.by_frame[f].box) for f in shared]
            rows.append({
                "id_a": a_id,
                "id_b": b_id,
                "relation": "coexisting" if shared else "interleaved",
                "shared_frames": len(shared),
                "first_shared_frame": shared[0] if shared else "",
                "last_shared_frame": shared[-1] if shared else "",
                "mean_iou": _fmt(sum(ious) / len(ious)) if ious else "",
                "max_iou": _fmt(max(ious)) if ious else "",
                "a_span": f"{a.first_frame}-{a.last_frame}",
                "b_span": f"{b.first_frame}-{b.last_frame}",
                "a_class_hist": _hist_str(a.class_hist),
                "b_class_hist": _hist_str(b.class_hist),
                "class_compat": class_compat(a.class_hist, b.class_hist),
                "decision": decisions.get(frozenset((a_id, b_id)), ""),
            })
    return rows


def _velocity(window: Sequence[Obs]) -> tuple[float, float] | None:
    if len(window) < 2:
        return None
    df = window[-1].frame - window[0].frame
    (x0, y0), (x1, y1) = window[0].center, window[-1].center
    return ((x1 - x0) / df, (y1 - y0) / df)


def find_merge_candidates(
    tracks: Mapping[int, Track],
    params: AuditParams,
    decisions: Mapping[tuple[int, int], str] | None = None,
) -> list[dict]:
    """Propose ordered links A -> B (A ends before B starts), with evidence."""
    decisions = decisions or {}
    ids = sorted(tracks)
    rows: list[dict] = []
    for a_id in ids:
        a = tracks[a_id]
        a_last = a.obs[-1]
        for b_id in ids:
            if a_id == b_id:
                continue
            b = tracks[b_id]
            if b.first_frame <= a.last_frame:
                continue
            frame_gap = b.first_frame - a.last_frame
            if frame_gap > params.max_frame_gap:
                continue
            b_first = b.obs[0]
            (ax, ay), (bx, by) = a_last.center, b_first.center
            dist = math.hypot(bx - ax, by - ay)
            scale = max(a_last.diag, b_first.diag) or 1.0
            norm_disp = dist / scale
            if norm_disp > params.max_norm_displacement:
                continue

            fwd = bwd = None
            if a.n_obs >= params.min_obs_for_motion:
                v = _velocity(a.obs[-params.velocity_window:])
                if v is not None:
                    px, py = ax + v[0] * frame_gap, ay + v[1] * frame_gap
                    fwd = math.hypot(bx - px, by - py) / scale
            if b.n_obs >= params.min_obs_for_motion:
                v = _velocity(b.obs[:params.velocity_window])
                if v is not None:
                    px, py = bx - v[0] * frame_gap, by - v[1] * frame_gap
                    bwd = math.hypot(ax - px, ay - py) / scale
            resids = [r for r in (fwd, bwd) if r is not None]
            if not resids:
                motion_flag = "insufficient"
            elif max(resids) <= params.motion_tolerance:
                motion_flag = "consistent"
            else:
                motion_flag = "inconsistent"

            max_area = max(_area(a_last.box), _area(b_first.box))
            size_ratio = (min(_area(a_last.box), _area(b_first.box)) / max_area
                          if max_area > 0 else 0.0)
            compat = class_compat(a.class_hist, b.class_hist)
            end_edge = near_edge(a_last.box, params)
            start_edge = near_edge(b_first.box, params)

            rows.append({
                "from_id": a_id,
                "to_id": b_id,
                "a_last_frame": a.last_frame,
                "b_first_frame": b.first_frame,
                "frame_gap": frame_gap,
                "missing_frames": frame_gap - 1,
                "a_n_obs": a.n_obs,
                "b_n_obs": b.n_obs,
                "a_class_hist": _hist_str(a.class_hist),
                "b_class_hist": _hist_str(b.class_hist),
                "class_compat": compat,
                "center_dist_px": _fmt(dist, 1),
                "norm_displacement": _fmt(norm_disp),
                "iou_last_first": _fmt(iou(a_last.box, b_first.box)),
                "size_ratio": _fmt(size_ratio),
                "motion_resid_fwd_norm": _fmt(fwd),
                "motion_resid_bwd_norm": _fmt(bwd),
                "motion_flag": motion_flag,
                "a_end_near_edge": _bool(end_edge),
                "b_start_near_edge": _bool(start_edge),
                "successor_candidates": 0,
                "predecessor_candidates": 0,
                "evidence_flags": "",
                "review_ref": (f"ids {a_id}->{b_id}; frames "
                               f"{a.last_frame} and {b.first_frame}"),
                "decision": decisions.get((a_id, b_id), ""),
                "_flags": {
                    "short_from_track": a.n_obs < params.min_obs_for_motion,
                    "short_to_track": b.n_obs < params.min_obs_for_motion,
                    "motion_insufficient": motion_flag == "insufficient",
                    "motion_inconsistent": motion_flag == "inconsistent",
                    "class_different": compat == "different",
                    "edge_adjacent_end": end_edge,
                    "edge_adjacent_start": start_edge,
                },
            })

    succ = Counter(r["from_id"] for r in rows)
    pred = Counter(r["to_id"] for r in rows)
    for r in rows:
        r["successor_candidates"] = succ[r["from_id"]]
        r["predecessor_candidates"] = pred[r["to_id"]]
        flags = [name for name, on in r.pop("_flags").items() if on]
        if r["successor_candidates"] > 1:
            flags.append("ambiguous_successor")
        if r["predecessor_candidates"] > 1:
            flags.append("ambiguous_predecessor")
        r["evidence_flags"] = ";".join(flags)
    return rows


def build_groups(
    tracks: Mapping[int, Track], canonical: Mapping[int, int]
) -> list[dict]:
    members: dict[int, list[int]] = defaultdict(list)
    for tid, cid in canonical.items():
        members[cid].append(tid)
    rows = []
    for cid in sorted(members):
        ids = sorted(members[cid], key=lambda t: (tracks[t].first_frame, t))
        hist: Counter = Counter()
        for tid in ids:
            hist.update(tracks[tid].class_hist)
        hist_d = dict(sorted(hist.items()))
        rows.append({
            "canonical_id": cid,
            "member_ids": "|".join(str(t) for t in ids),
            "n_members": len(ids),
            "first_frame": min(tracks[t].first_frame for t in ids),
            "last_frame": max(tracks[t].last_frame for t in ids),
            "n_obs": sum(tracks[t].n_obs for t in ids),
            "class_hist": _hist_str(hist_d),
            "modal_class": _modal(hist_d),
            "class_conflict": _bool(len(hist_d) > 1),
            "source": "approved_mapping" if len(ids) > 1 else "identity",
        })
    return rows


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def _check_output_location(raw: Path, out: Path) -> None:
    raw_dir = raw.parent
    if out == raw_dir or raw_dir in out.parents:
        raise AuditError("output directory must not be inside the raw log directory")
    if out.exists():
        raise AuditError(f"output directory already exists: {out}")


def run_audit(
    raw_log: str | Path,
    out_dir: str | Path,
    approvals_path: str | Path | None = None,
    params: AuditParams | None = None,
    expect_sha256: str | None = None,
) -> dict:
    """Run the audit. Reads the raw log; writes only inside ``out_dir``."""
    params = params or AuditParams()
    raw = Path(raw_log).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    if not raw.is_file():
        raise AuditError(f"raw log not found: {raw}")
    _check_output_location(raw, out)

    raw_sha = sha256_file(raw)
    sidecar = verify_sidecar(raw, raw_sha, expect_sha256)
    tracks = load_track_log(raw)
    if not tracks:
        raise AuditError("raw log has no observations")

    approvals: list[Approval] = []
    approvals_bytes: bytes | None = None
    if approvals_path is not None:
        ap = Path(approvals_path).expanduser().resolve()
        if not ap.is_file():
            raise ApprovalError(f"approvals file not found: {ap}")
        approvals_bytes = ap.read_bytes()
        approvals = load_approvals(ap, tracks)

    ordered = {(a.from_id, a.to_id): a.decision for a in approvals}
    unordered = {frozenset((a.from_id, a.to_id)): a.decision for a in approvals}

    summary = summarize_tracks(tracks, params)
    overlaps = find_overlaps(tracks, unordered)
    merges = find_merge_candidates(tracks, params, ordered)
    canonical = build_identity_map(tracks, approvals)
    groups = build_groups(tracks, canonical)
    group_size = Counter(canonical.values())
    id_map = [{
        "track_id": tid,
        "canonical_id": canonical[tid],
        "group_size": group_size[canonical[tid]],
        "source": "approved_mapping" if group_size[canonical[tid]] > 1 else "identity",
    } for tid in sorted(tracks)]

    candidate_pairs = {(r["from_id"], r["to_id"]) for r in merges}
    all_frames = [o.frame for t in tracks.values() for o in t.obs]
    report = {
        "schema_version": SCHEMA_VERSION,
        "raw_log": {
            "path": str(raw),
            "sha256": raw_sha,
            "bytes": raw.stat().st_size,
            "rows": len(all_frames),
            "tracks": len(tracks),
            "first_frame": min(all_frames),
            "last_frame": max(all_frames),
            **sidecar,
        },
        "approvals": None if approvals_bytes is None else {
            "sha256": hashlib.sha256(approvals_bytes).hexdigest(),
            "snapshot": APPROVALS_SNAPSHOT,
            "entries": [
                {"from_id": a.from_id, "to_id": a.to_id, "decision": a.decision,
                 "reviewer": a.reviewer, "evidence_note": a.evidence_note}
                for a in approvals],
            "approved_not_in_candidates": [
                f"{a.from_id}->{a.to_id}" for a in approvals
                if a.decision == "approved"
                and (a.from_id, a.to_id) not in candidate_pairs],
        },
        "params": asdict(params),
        "counts": {
            "tracks": len(tracks),
            "short_tracks": sum(r["short_track"] == "true" for r in summary),
            "overlap_pairs": len(overlaps),
            "coexisting_pairs": sum(r["relation"] == "coexisting" for r in overlaps),
            "merge_candidates": len(merges),
            "ambiguous_candidates": sum(
                "ambiguous" in r["evidence_flags"] for r in merges),
            "canonical_identities": len(groups),
            "mapped_tracks": sum(1 for r in id_map if r["source"] != "identity"),
        },
        "notes": [
            "merge_proposals.csv rows are proposals for human review, not decisions",
            "identity_map.csv applies only approved mappings, in this analysis layer",
            "the raw log is never modified",
        ],
    }

    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "track_summary.csv", SUMMARY_COLUMNS, summary)
    _write_csv(out / "overlap_pairs.csv", OVERLAP_COLUMNS, overlaps)
    _write_csv(out / "merge_proposals.csv", MERGE_COLUMNS, merges)
    _write_csv(out / "identity_map.csv", MAP_COLUMNS, id_map)
    _write_csv(out / "identity_groups.csv", GROUP_COLUMNS, groups)
    if approvals_bytes is not None:
        with (out / APPROVALS_SNAPSHOT).open("xb") as handle:
            handle.write(approvals_bytes)
    with (out / "audit_report.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if sha256_file(raw) != raw_sha:  # pragma: no cover - defensive
        raise AuditError("raw log changed while the audit ran")
    return report


def _build_parser() -> argparse.ArgumentParser:
    defaults = AuditParams()
    parser = argparse.ArgumentParser(
        prog="python -m traffic_analysis.identity_audit",
        description="Audit track-ID fragmentation in a frozen raw track log.")
    parser.add_argument("--raw-log", required=True, help="path to the frozen track_log.csv")
    parser.add_argument("--approvals", help="human-authored approved_mappings.csv")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out-dir", help="new output directory (must not exist)")
    target.add_argument("--out-root", help="create analysis-<timestamp>/ under this directory")
    parser.add_argument("--expect-sha256", help="refuse to run unless the raw log has this SHA-256")
    parser.add_argument("--frame-width", type=int, default=defaults.frame_width)
    parser.add_argument("--frame-height", type=int, default=defaults.frame_height)
    parser.add_argument("--edge-margin", type=float, default=defaults.edge_margin)
    parser.add_argument("--max-frame-gap", type=int, default=defaults.max_frame_gap)
    parser.add_argument("--max-norm-displacement", type=float,
                        default=defaults.max_norm_displacement)
    parser.add_argument("--min-obs-for-motion", type=int, default=defaults.min_obs_for_motion)
    parser.add_argument("--velocity-window", type=int, default=defaults.velocity_window)
    parser.add_argument("--motion-tolerance", type=float, default=defaults.motion_tolerance)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        params = AuditParams(
            frame_width=args.frame_width, frame_height=args.frame_height,
            edge_margin=args.edge_margin, max_frame_gap=args.max_frame_gap,
            max_norm_displacement=args.max_norm_displacement,
            min_obs_for_motion=args.min_obs_for_motion,
            velocity_window=args.velocity_window,
            motion_tolerance=args.motion_tolerance)
    except ValueError as exc:
        print(f"audit failed: invalid parameter: {exc}", file=sys.stderr)
        return 2
    if args.out_dir:
        out = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path(args.out_root).expanduser() / f"analysis-{stamp}"
    try:
        report = run_audit(args.raw_log, out, args.approvals, params,
                           args.expect_sha256)
    except AuditError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"output: {out.expanduser().resolve()}")
    print(f"raw log sha256: {report['raw_log']['sha256']} "
          f"(sidecar: {report['raw_log']['status']})")
    for key, value in report["counts"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
