import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.analytics.exporter import (
    export_stats_csv,
    export_stats_json,
    prepare_stats_for_export,
)


class TestAnalyticsExporter(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "total": 12,
            "by_class": {"CAR": 8, "BIKE": 4},
            "by_direction": {"UP": 5, "DOWN": 7},
            "density": "MEDIUM",
            "wrong_way": 2,
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)

    def test_json_export_preserves_stats(self):
        path = self.directory / "nested" / "stats.json"

        export_stats_json(self.stats, path)

        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.stats)

    def test_csv_export_contains_all_metrics(self):
        path = self.directory / "nested" / "stats.csv"

        export_stats_csv(self.stats, path)

        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))

        self.assertEqual(rows[0], ["metric", "value"])
        values = {row[0]: row[1] for row in rows[1:]}
        self.assertEqual(values["total"], "12")
        self.assertEqual(json.loads(values["by_class"]), {"CAR": 8, "BIKE": 4})
        self.assertEqual(json.loads(values["by_direction"]), {"UP": 5, "DOWN": 7})
        self.assertEqual(values["density"], "MEDIUM")
        self.assertEqual(values["wrong_way"], "2")

    def test_missing_optional_metrics_use_defaults(self):
        path = self.directory / "minimal.csv"

        export_stats_csv({"total": 3}, path)

        with path.open(encoding="utf-8-sig", newline="") as file:
            values = {row[0]: row[1] for row in list(csv.reader(file))[1:]}

        self.assertEqual(values["total"], "3")
        self.assertEqual(values["density"], "UNKNOWN")
        self.assertEqual(values["wrong_way"], "0")

    def test_prepare_dashboard_stats_for_export_preserves_unavailable_directions(self):
        stats = {
            "total": 12,
            "Car": 8,
            "Bike": 4,
            "Bus": 0,
            "Truck": 0,
            "density": "MEDIUM",
            "wrong_way": 2,
        }

        export_stats = prepare_stats_for_export(stats)

        self.assertEqual(
            export_stats["by_class"],
            {"Car": 8, "Bike": 4, "Bus": 0, "Truck": 0},
        )
        self.assertNotIn("by_direction", export_stats)
        self.assertNotIn("by_class", stats)

        csv_path = self.directory / "dashboard.csv"
        json_path = self.directory / "dashboard.json"
        export_stats_csv(export_stats, csv_path)
        export_stats_json(export_stats, json_path)

        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            csv_values = {row[0]: row[1] for row in list(csv.reader(file))[1:]}
        self.assertEqual(json.loads(csv_values["by_direction"]), {})
        self.assertNotIn("by_direction", json.loads(json_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
