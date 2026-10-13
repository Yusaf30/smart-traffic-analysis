"""Resolve a track's class from confidence-weighted pre-crossing evidence."""
from collections.abc import Iterable, Mapping
from math import isfinite


UNCERTAIN_CLASS = "Uncertain"
MIN_OBSERVATIONS = 3
MIN_CONFIDENCE = 0.40
MIN_CLASS_SUPPORT = 0.70

ClassObservation = tuple[int, int, float]


def resolve_track_class(
    observations: Iterable[ClassObservation],
    crossing_frame: int,
    class_names: Mapping[int, str],
) -> str:
    """Resolve class from (frame, class_id, confidence) observations before crossing.

    Unknown class IDs and invalid confidences are ignored. Insufficient or
    conflicting evidence returns ``Uncertain``. This function does not handle
    track identity or canonical ID mapping.
    """
    evidence: dict[str, float] = {}
    eligible_count = 0

    for frame, class_id, confidence in observations:
        if frame >= crossing_frame or class_id not in class_names:
            continue
        if (not isfinite(confidence)
                or not MIN_CONFIDENCE <= confidence <= 1.0):
            continue

        class_name = class_names[class_id]
        evidence[class_name] = evidence.get(class_name, 0.0) + confidence
        eligible_count += 1

    if eligible_count < MIN_OBSERVATIONS or not evidence:
        return UNCERTAIN_CLASS

    total_weight = sum(evidence.values())
    leading_class = max(evidence, key=evidence.get)
    if evidence[leading_class] / total_weight >= MIN_CLASS_SUPPORT:
        return leading_class
    return UNCERTAIN_CLASS
