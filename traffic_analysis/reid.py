"""Conservative vehicle identity matching for unique-vehicle counting.

Maps ByteTrack track IDs to stable canonical vehicle IDs using appearance +
motion compatibility, without merging ambiguous candidates.

All thresholds are configurable experimental defaults, not calibrated values.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configurable experimental defaults (not calibrated thresholds)
# ---------------------------------------------------------------------------

REID_ENABLED: bool = True
REID_MAX_GAP_SECONDS: float = 1.0          # max seconds between observations
REID_MAX_SPEED_PX_PER_SEC: float = 200.0   # max plausible speed for motion gate
REID_HIST_THRESHOLD: float = 0.70          # min spatial HSV histogram similarity
REID_HIST_BINS: tuple[int, int, int] = (8, 8, 8)


@dataclass
class Appearance:
    """Visual appearance sample extracted from a vehicle crop."""

    class_id: int
    hist: np.ndarray                       # normalized HSV histogram
    centroid: tuple[float, float]          # (cx, cy) in pixels
    frame_id: int


@dataclass(frozen=True)
class MatchDiagnostic:
    """Decision details for one raw tracker observation."""

    track_id: int
    canonical_id: int
    matched_existing: bool
    similarity: Optional[float]
    reason: str


@dataclass
class CanonicalRecord:
    """One canonical vehicle: its ID, appearance history, last observation.

    ``registered_class_id`` is the class used by the hard matching gate. It is
    set when the canonical is created and never overwritten by a single later
    observation, so a class flip cannot break identity matching. All observed
    classes are appended to ``class_history`` as evidence.
    """

    canonical_id: int
    registered_class_id: int
    class_history: list[int] = field(default_factory=list)
    appearances: list[Appearance] = field(default_factory=list)
    last_frame: int = -1
    last_centroid: Optional[tuple[float, float]] = None


def extract_appearance(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    class_id: int,
    frame_id: int,
    hist_bins: tuple[int, int, int] = REID_HIST_BINS,
) -> Appearance:
    """Extract a normalized 2x2 spatial HSV histogram from a valid crop.

    Completely out-of-frame or degenerate boxes return a zero histogram.
    Partially out-of-frame boxes are clipped to the image boundaries.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]

    # Reject malformed boxes and boxes that do not intersect the image.
    if (
        x2 <= x1
        or y2 <= y1
        or x2 <= 0
        or y2 <= 0
        or x1 >= w
        or y1 >= h
    ):
        return Appearance(
            class_id=class_id,
            hist=np.zeros(int(np.prod(hist_bins)), dtype=np.float32),
            centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            frame_id=frame_id,
        )

    # Clip partially visible boxes to the image boundaries.
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        hist = np.zeros(int(np.prod(hist_bins)), dtype=np.float32)
        return Appearance(
            class_id=class_id,
            hist=hist,
            centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            frame_id=frame_id,
        )

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    cell_histograms = []
    for row in range(2):
        for column in range(2):
            cell = hsv[
                row * hsv.shape[0] // 2:(row + 1) * hsv.shape[0] // 2,
                column * hsv.shape[1] // 2:(column + 1) * hsv.shape[1] // 2,
            ]
            if cell.size == 0:
                cell_histograms.append(
                    np.zeros(int(np.prod(hist_bins)), dtype=np.float32)
                )
                continue
            hist = cv2.calcHist(
                [cell], [0, 1, 2], None, list(hist_bins),
                [0, 180, 0, 256, 0, 256],
            )
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)
            cell_histograms.append(hist.flatten())
    hist = np.concatenate(cell_histograms)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)

    return Appearance(
        class_id=class_id,
        hist=hist.flatten(),
        centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        frame_id=frame_id,
    )

def appearance_similarity(a: Appearance, b: Appearance) -> float:
    """Cosine-like similarity between two normalized histograms.

    Returns a value in [0, 1]; higher means more similar.
    """
    denom = float(np.linalg.norm(a.hist) * np.linalg.norm(b.hist))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a.hist, b.hist) / denom)


def class_compatible(class_id_a: int, class_id_b: int) -> bool:
    """Hard gate: only same vehicle class can match."""
    return int(class_id_a) == int(class_id_b)


def motion_plausible(
    prev: CanonicalRecord,
    curr_centroid: tuple[float, float],
    curr_frame: int,
    fps: float,
    max_gap_seconds: float = REID_MAX_GAP_SECONDS,
    max_speed_px_per_sec: float = REID_MAX_SPEED_PX_PER_SEC,
) -> bool:
    """Check temporal gap and displacement against a speed-based bound.

    Does NOT require IoU overlap. Uses elapsed frames / fps to convert a
    speed limit (px/s) into a max displacement for the actual gap.
    """
    if prev.last_frame < 0:
        return True  # first observation of this canonical

    gap_frames = curr_frame - prev.last_frame
    if gap_frames <= 0:
        return True  # same frame; no temporal constraint
    if fps <= 0:
        fps = 25.0

    max_gap_frames = max_gap_seconds * fps
    if gap_frames > max_gap_frames:
        return False

    if prev.last_centroid is None:
        return True

    dx = curr_centroid[0] - prev.last_centroid[0]
    dy = curr_centroid[1] - prev.last_centroid[1]
    displacement = (dx * dx + dy * dy) ** 0.5
    max_displacement = max_speed_px_per_sec * (gap_frames / fps)
    return displacement <= max_displacement


class CanonicalCounter:
    """Maps track IDs to canonical vehicle IDs conservatively.

    Design rules:
    - A track ID always resolves to the same canonical ID once assigned.
    - Repeated observations of a track ID update its existing canonical record.
    - Two tracks active in the same frame can never share a canonical ID.
    - Ambiguous matches create a new canonical ID (fragmentation preferred).
    - Detections without tracker IDs are never registered here.
    """

    def __init__(
        self,
        *,
        enabled: bool = REID_ENABLED,
        max_gap_seconds: float = REID_MAX_GAP_SECONDS,
        max_speed_px_per_sec: float = REID_MAX_SPEED_PX_PER_SEC,
        hist_threshold: float = REID_HIST_THRESHOLD,
        fps: float = 25.0,
    ) -> None:
        self.enabled = enabled
        self.max_gap_seconds = max_gap_seconds
        self.max_speed_px_per_sec = max_speed_px_per_sec
        self.hist_threshold = hist_threshold
        self.fps = fps

        self._track_to_canonical: dict[int, int] = {}
        self._canonicals: dict[int, CanonicalRecord] = {}
        self._next_canonical_id: int = 1
        # Canonical IDs already claimed within the current frame (prevents two
        # simultaneous tracks from sharing one canonical ID).
        self._claimed_this_frame: set[int] = set()
        self._current_frame_id: Optional[int] = None
        self._frame_diagnostics: list[MatchDiagnostic] = []

    def begin_frame(self, frame_id: int) -> None:
        """Reset per-frame canonical claims when the frame id advances."""
        if self._current_frame_id != frame_id:
            self._claimed_this_frame.clear()
            self._current_frame_id = frame_id

    def _appearance_or_none(
        self,
        frame: Optional[np.ndarray],
        bbox: tuple[float, float, float, float],
        class_id: int,
        frame_id: int,
    ) -> Optional[Appearance]:
        """Return an Appearance if a frame is available, else None.

        Missing appearance evidence must never be used to manufacture a ReID
        match; callers treat ``None`` as "no appearance comparison possible".
        """
        if frame is None:
            return None
        return extract_appearance(frame, bbox, class_id, frame_id)

    def _update_existing(
        self,
        track_id: int,
        canonical_id: int,
        class_id: int,
        centroid: tuple[float, float],
        frame_id: int,
        app: Optional[Appearance],
    ) -> None:
        """Refresh an already-mapped track's canonical record.

        The mapping is preserved. The registered class (hard gate) is immutable;
        a class flip is recorded in history only.
        """
        rec = self._canonicals[canonical_id]
        if app is not None:
            rec.appearances.append(app)
        rec.last_frame = frame_id
        rec.last_centroid = centroid
        rec.class_history.append(class_id)

    def _match_or_new(
        self,
        track_id: int,
        class_id: int,
        bbox: tuple[float, float, float, float],
        centroid: tuple[float, float],
        frame_id: int,
        app: Optional[Appearance],
    ) -> int:
        """Match a new track against unclaimed canonicals, else allocate new."""

        # Candidate canonicals compatible by class and motion, and not already
        # claimed by another track in this frame.
        candidates: list[tuple[int, float]] = []  # (canonical_id, similarity)
        class_candidate_count = 0
        motion_candidate_count = 0
        appearance_comparison_count = 0
        claimed_class_candidate_count = 0
        appearance_is_valid = app is not None and np.linalg.norm(app.hist) > 1e-12
        if self.enabled:
            for canonical_id, rec in self._canonicals.items():
                if not class_compatible(rec.registered_class_id, class_id):
                    continue
                if canonical_id in self._claimed_this_frame:
                    claimed_class_candidate_count += 1
                    continue
                class_candidate_count += 1
                if not motion_plausible(
                    rec,
                    centroid,
                    frame_id,
                    self.fps,
                    self.max_gap_seconds,
                    self.max_speed_px_per_sec,
                ):
                    continue
                motion_candidate_count += 1

                # Appearance similarity requires real histogram evidence on
                # both sides. Missing evidence cannot manufacture a match.
                if app is None or not appearance_is_valid or not rec.appearances:
                    continue
                appearance_comparison_count += 1
                sim = appearance_similarity(app, rec.appearances[-1])
                if sim >= self.hist_threshold:
                    candidates.append((canonical_id, sim))

        if not candidates:
            # No match: new canonical ID.
            canonical_id = self._next_canonical_id
            self._next_canonical_id += 1
            self._canonicals[canonical_id] = CanonicalRecord(
                canonical_id=canonical_id,
                registered_class_id=class_id,
                class_history=[],
                last_frame=frame_id,
                last_centroid=centroid,
            )
            matched_existing = False
            similarity = None
            if not self.enabled:
                reason = "reidentification_disabled"
            elif app is None:
                reason = "missing_appearance"
            elif not appearance_is_valid:
                reason = "invalid_appearance"
            elif not class_candidate_count and claimed_class_candidate_count:
                reason = "canonical_claimed_this_frame"
            elif not class_candidate_count:
                reason = "no_class_compatible_candidate"
            elif not motion_candidate_count:
                reason = "motion_or_gap_rejected"
            elif not appearance_comparison_count:
                reason = "no_appearance_history"
            else:
                reason = "similarity_below_threshold"
        elif len(candidates) == 1:
            canonical_id = candidates[0][0]
            matched_existing = True
            similarity = candidates[0][1]
            reason = "class_motion_appearance_match"
        else:
            # Ambiguous: multiple canonicals match. Pick the best only if it
            # clearly beats the runner-up; otherwise fragment (new ID).
            candidates.sort(key=lambda c: c[1], reverse=True)
            best, runner_up = candidates[0], candidates[1]
            margin = best[1] - runner_up[1]
            if margin >= 0.05:
                canonical_id = best[0]
                matched_existing = True
                similarity = best[1]
                reason = "best_candidate_clear_margin"
            else:
                canonical_id = self._next_canonical_id
                self._next_canonical_id += 1
                self._canonicals[canonical_id] = CanonicalRecord(
                    canonical_id=canonical_id,
                    registered_class_id=class_id,
                    class_history=[],
                    last_frame=frame_id,
                    last_centroid=centroid,
                )
                matched_existing = False
                similarity = best[1]
                reason = "ambiguous_candidates"

        self._track_to_canonical[track_id] = canonical_id
        self._claimed_this_frame.add(canonical_id)

        rec = self._canonicals.get(canonical_id)
        if rec is None:
            rec = CanonicalRecord(
                canonical_id=canonical_id,
                registered_class_id=class_id,
                class_history=[class_id],
                last_frame=frame_id,
                last_centroid=centroid,
            )
            self._canonicals[canonical_id] = rec
        if app is not None:
            rec.appearances.append(app)
        rec.last_frame = frame_id
        rec.last_centroid = centroid
        rec.class_history.append(class_id)
        self._frame_diagnostics.append(
            MatchDiagnostic(
                track_id=track_id,
                canonical_id=canonical_id,
                matched_existing=matched_existing,
                similarity=similarity,
                reason=reason,
            )
        )
        return canonical_id

    def register_frame(
        self,
        observations: list[
            tuple[int, int, tuple[float, float, float, float], tuple[float, float]]
        ],
        frame_id: int,
        frame: Optional[np.ndarray] = None,
    ) -> dict[int, int]:
        """Batch-register all tracks observed in a single frame.

        ``observations`` is a list of ``(track_id, class_id, bbox, centroid)``
        tuples.

        Two-pass ordering guarantees same-frame canonical uniqueness regardless
        of call order:
        1. Already-mapped tracks reserve their canonical IDs first.
        2. New tracks match only against canonicals not already claimed this
           frame.

        Returns ``{track_id: canonical_id}``.
        """
        self.begin_frame(frame_id)
        self._frame_diagnostics = []
        results: dict[int, int] = {}

        # Pass 1: reserve canonicals for already-mapped tracks.
        for track_id, class_id, bbox, centroid in observations:
            track_id = int(track_id)
            class_id = int(class_id)
            if track_id in self._track_to_canonical:
                canonical_id = self._track_to_canonical[track_id]
                self._claimed_this_frame.add(canonical_id)
                app = self._appearance_or_none(frame, bbox, class_id, frame_id)
                self._update_existing(
                    track_id, canonical_id, class_id, centroid, frame_id, app
                )
                results[track_id] = canonical_id
                self._frame_diagnostics.append(
                    MatchDiagnostic(
                        track_id=track_id,
                        canonical_id=canonical_id,
                        matched_existing=False,
                        similarity=None,
                        reason="existing_track_mapping",
                    )
                )

        # Pass 2: match new tracks against unclaimed canonicals.
        for track_id, class_id, bbox, centroid in observations:
            track_id = int(track_id)
            class_id = int(class_id)
            if track_id in results:
                continue
            app = self._appearance_or_none(frame, bbox, class_id, frame_id)
            canonical_id = self._match_or_new(
                track_id, class_id, bbox, centroid, frame_id, app
            )
            results[track_id] = canonical_id

        return results

    def get_frame_diagnostics(self) -> tuple[MatchDiagnostic, ...]:
        """Return decisions from the most recent ``register_frame`` call."""
        return tuple(self._frame_diagnostics)

    def register(
        self,
        track_id: int,
        class_id: int,
        bbox: tuple[float, float, float, float],
        centroid: tuple[float, float],
        frame_id: int,
        frame: Optional[np.ndarray] = None,
    ) -> int:
        """Register a single track observation (convenience over register_frame).

        Same-frame uniqueness is enforced across multiple ``register()`` calls
        for the same frame_id via per-frame claim tracking.
        """
        return self.register_frame(
            [(track_id, class_id, bbox, centroid)], frame_id, frame
        )[track_id]

    def get_unique_count(self) -> int:
        """Cumulative number of distinct canonical vehicles seen so far."""
        return len(self._canonicals)

    def get_canonical_id(self, track_id: int) -> Optional[int]:
        """Return the canonical ID for a track ID, or None if unseen."""
        return self._track_to_canonical.get(int(track_id))
