from traffic_analysis.crossing_validation import (
    CrossingObservation,
    validate_crossing,
)


def observation(frame, center_y, width=100, height=100, center_x=100):
    half_width, half_height = width / 2, height / 2
    box = (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )
    return CrossingObservation(frame, (center_x, center_y), box)


def test_consecutive_vehicle_crossing_is_confirmed_once_eligible():
    result = validate_crossing(
        observation(10, 320), observation(11, 340), line_y=330
    )

    assert result.crossed and result.confirmed and result.reason is None


def test_one_missing_frame_can_still_confirm_normal_crossing():
    result = validate_crossing(
        observation(10, 315), observation(12, 345), line_y=330
    )

    assert result.crossed and result.confirmed


def test_large_detection_gap_is_unconfirmed():
    result = validate_crossing(
        observation(10, 374), observation(19, 265), line_y=330
    )

    assert result.crossed and not result.confirmed
    assert result.reason == "detection_gap"


def test_abrupt_center_jump_is_unconfirmed():
    result = validate_crossing(
        observation(10, 300), observation(11, 380), line_y=330
    )

    assert result.crossed and not result.confirmed
    assert result.reason == "center_jump"


def test_abrupt_box_size_change_is_unconfirmed():
    result = validate_crossing(
        observation(10, 320),
        observation(11, 340, width=200, height=200),
        line_y=330,
    )

    assert result.crossed and not result.confirmed
    assert result.reason == "box_size_jump"


def test_transition_without_line_crossing_is_not_a_candidate():
    result = validate_crossing(
        observation(10, 310), observation(11, 320), line_y=330
    )

    assert not result.crossed and not result.confirmed
    assert result.reason is None


# --- Axis support: vertical (X) counting line for horizontally-flowing traffic ---

def x_observation(frame, center_x, width=100, height=100, center_y=450):
    half_width, half_height = width / 2, height / 2
    box = (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )
    return CrossingObservation(frame, (center_x, center_y), box)


def test_vertical_line_crossing_confirmed_both_directions():
    # Left-to-right across a vertical line at x=640. Box diagonal (~141) with a
    # ~28px/frame step keeps normalized motion under the 0.25 threshold.
    ltr = validate_crossing(
        x_observation(10, 620, width=120, height=80),
        x_observation(11, 648, width=120, height=80),
        line_y=640,
        axis="x",
    )
    assert ltr.crossed and ltr.confirmed and ltr.reason is None

    # Right-to-left across the same vertical line.
    rtl = validate_crossing(
        x_observation(10, 660, width=120, height=80),
        x_observation(11, 632, width=120, height=80),
        line_y=640,
        axis="x",
    )
    assert rtl.crossed and rtl.confirmed and rtl.reason is None


def test_vertical_line_no_crossing_when_staying_on_one_side():
    # Vehicle travels horizontally but stays left of the line.
    result = validate_crossing(
        x_observation(10, 500), x_observation(11, 600), line_y=640, axis="x"
    )
    assert not result.crossed and not result.confirmed


def test_vertical_line_genuine_crossing_across_supported_gap():
    # One missing frame (gap=2) still confirms a normal, proportional move.
    result = validate_crossing(
        x_observation(10, 615, width=140, height=90),
        x_observation(12, 665, width=140, height=90),
        line_y=640,
        axis="x",
    )
    assert result.crossed and result.confirmed


def test_vertical_line_large_gap_is_unconfirmed():
    result = validate_crossing(
        x_observation(10, 600), x_observation(15, 680), line_y=640, axis="x"
    )
    assert result.crossed and not result.confirmed
    assert result.reason == "detection_gap"


def test_vertical_line_box_size_jump_is_unconfirmed():
    # Contaminated box that suddenly balloons while crossing the line.
    result = validate_crossing(
        x_observation(10, 620),
        x_observation(11, 660, width=300, height=300),
        line_y=640,
        axis="x",
    )
    assert result.crossed and not result.confirmed
    assert result.reason == "box_size_jump"


def test_vertical_line_center_jump_is_unconfirmed():
    # Implausible horizontal teleport across the line within one frame.
    result = validate_crossing(
        x_observation(10, 620), x_observation(11, 900), line_y=640, axis="x"
    )
    assert result.crossed and not result.confirmed
    assert result.reason == "center_jump"


def test_invalid_axis_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        validate_crossing(
            observation(10, 320), observation(11, 340), line_y=330, axis="z"
        )


def test_id47_false_crossing_pattern_is_not_counted():
    # Real india_trafic track 47 sits at the right edge (cx ~1045-1050) and
    # never approaches the x=640 counting line, so it must not register as a
    # crossing on the vertical line.
    prev = x_observation(240, 1050, width=45, height=103, center_y=449)
    curr = x_observation(241, 1048, width=42, height=95, center_y=451)
    result = validate_crossing(prev, curr, line_y=640, axis="x")
    assert not result.crossed and not result.confirmed

    # It also never crossed the legacy horizontal line at y=330.
    result_y = validate_crossing(prev, curr, line_y=330, axis="y")
    assert not result_y.crossed and not result_y.confirmed