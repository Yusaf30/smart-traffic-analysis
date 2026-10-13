"""Validate whether a tracked box transition is sufficient to confirm a crossing."""
from dataclasses import dataclass
from math import hypot, isfinite


MAX_CROSSING_GAP_FRAMES = 2
MAX_CENTER_JUMP_DIAGONALS_PER_FRAME = 0.25
MAX_BOX_DIMENSION_RATIO = 1.5


@dataclass(frozen=True)
class CrossingObservation:
    frame: int
    center: tuple[float, float]
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CrossingValidation:
    crossed: bool
    confirmed: bool
    reason: str | None = None


def validate_crossing(
    previous: CrossingObservation,
    current: CrossingObservation,
    line_y: float,
    axis: str = "y",
) -> CrossingValidation:
    """Confirm a center crossing only across temporally and geometrically plausible boxes.

    ``line_y`` is the counting-line coordinate measured along ``axis``:
    the image Y for ``axis="y"`` (a horizontal line) or the image X for
    ``axis="x"`` (a vertical line). All geometric plausibility checks
    (frame gap, box presence, box-size ratio, and full 2-D center-jump) are
    applied identically for both axes, so an implausible jump or a
    contaminated box is rejected the same way regardless of line orientation.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    coord = 1 if axis == "y" else 0
    previous_c = previous.center[coord]
    current_c = current.center[coord]
    crossed = (previous_c < line_y <= current_c) or (
        previous_c >= line_y > current_c
    )
    if not crossed:
        return CrossingValidation(False, False)

    previous_y = previous.center[1]
    current_y = current.center[1]

    gap = current.frame - previous.frame
    if gap <= 0:
        return CrossingValidation(True, False, "invalid_frame_order")
    if gap > MAX_CROSSING_GAP_FRAMES:
        return CrossingValidation(True, False, "detection_gap")

    if previous.box is None and current.box is None:
        return CrossingValidation(True, True)
    if previous.box is None or current.box is None:
        return CrossingValidation(True, False, "missing_box")

    previous_width, previous_height = _dimensions(previous.box)
    current_width, current_height = _dimensions(current.box)
    if min(previous_width, previous_height, current_width, current_height) <= 0:
        return CrossingValidation(True, False, "invalid_box")
    if not all(isfinite(value) for value in (*previous.box, *current.box)):
        return CrossingValidation(True, False, "invalid_box")

    dimension_ratio = max(
        current_width / previous_width,
        previous_width / current_width,
        current_height / previous_height,
        previous_height / current_height,
    )
    if dimension_ratio > MAX_BOX_DIMENSION_RATIO:
        return CrossingValidation(True, False, "box_size_jump")

    previous_diagonal = hypot(previous_width, previous_height)
    current_diagonal = hypot(current_width, current_height)
    mean_diagonal = (previous_diagonal + current_diagonal) / 2
    dx = current.center[0] - previous.center[0]
    dy = current_y - previous_y
    normalized_motion = hypot(dx, dy) / (mean_diagonal * gap)
    if normalized_motion > MAX_CENTER_JUMP_DIAGONALS_PER_FRAME:
        return CrossingValidation(True, False, "center_jump")

    return CrossingValidation(True, True)


def _dimensions(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return x2 - x1, y2 - y1