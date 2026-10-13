import csv

import numpy as np
import pytest

from traffic_analysis.track_log import TrackLogger


HEADER = ["frame", "track_id", "class_id", "conf", "x1", "y1", "x2", "y2"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.reader(csv_file))


def test_writes_exact_header_and_row_values(tmp_path):
    path = tmp_path / "track_log.csv"
    logger = TrackLogger(path)
    logger.write(1, 42, 2, 0.987654321, (10, 20, 30, 40))
    logger.close()

    assert read_csv(path) == [
        HEADER,
        ["1", "42", "2", "0.987654321", "10.0", "20.0", "30.0", "40.0"],
    ]


def test_multiple_rows_keep_write_order(tmp_path):
    path = tmp_path / "track_log.csv"
    logger = TrackLogger(path)
    logger.write(2, 9, 3, 0.8, (1, 2, 3, 4))
    logger.write(3, 4, 5, 0.7, (5, 6, 7, 8))
    logger.close()

    rows = read_csv(path)
    assert rows[0] == HEADER
    assert rows[1][0:3] == ["2", "9", "3"]
    assert rows[2][0:3] == ["3", "4", "5"]
    assert len(rows) == 3


def test_preserves_float_precision_and_accepts_numpy_values(tmp_path):
    path = tmp_path / "track_log.csv"
    confidence = np.float64(0.12345678912345678)
    bbox = np.array([1.2345678901234567, 2.5, 30.75, 40.125])

    logger = TrackLogger(path)
    logger.write(
        np.int64(7),
        np.int32(8),
        np.int64(2),
        confidence,
        bbox,
    )
    logger.close()

    row = read_csv(path)[1]
    assert row[:3] == ["7", "8", "2"]
    assert row[3] == str(float(confidence))
    assert row[4:] == [str(float(value)) for value in bbox]
    assert float(row[3]) == float(confidence)
    assert float(row[4]) == float(bbox[0])


def test_empty_log_contains_only_one_header(tmp_path):
    path = tmp_path / "track_log.csv"
    logger = TrackLogger(path)
    logger.close()

    assert path.read_text(encoding="utf-8") == ",".join(HEADER) + "\n"


def test_close_is_idempotent_and_later_writes_fail(tmp_path):
    logger = TrackLogger(tmp_path / "track_log.csv")
    logger.close()
    logger.close()

    with pytest.raises(ValueError, match="closed TrackLogger"):
        logger.write(1, 1, 2, 0.9, (1, 2, 3, 4))


def test_refuses_to_overwrite_existing_file(tmp_path):
    path = tmp_path / "track_log.csv"
    original = "keep this file intact\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(FileExistsError):
        TrackLogger(path)

    assert path.read_text(encoding="utf-8") == original


def test_rejects_bbox_with_wrong_number_of_coordinates(tmp_path):
    logger = TrackLogger(tmp_path / "track_log.csv")
    try:
        with pytest.raises(ValueError, match="exactly four coordinates"):
            logger.write(1, 2, 3, 0.9, (1, 2, 3))
    finally:
        logger.close()