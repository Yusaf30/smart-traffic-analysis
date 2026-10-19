import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.ipc as arrow_ipc
from streamlit.testing.v1 import AppTest

from traffic_analysis.run_history import register_completed_run


SOURCE_APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
SOURCE_PROJECT = SOURCE_APP.parents[1]


class TestDashboardExports(unittest.TestCase):
    def run_dashboard(
        self,
        stats_text=None,
        events_text=None,
        results_dir=None,
        use_history=False,
    ):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        project_root = Path(temp_dir.name)
        dashboard_dir = project_root / "dashboard"
        dashboard_dir.mkdir()
        app_path = dashboard_dir / "app.py"
        app_path.write_text(SOURCE_APP.read_text(encoding="utf-8-sig"), encoding="utf-8")
        self.copy_traffic_analysis(project_root)
        selected_dir = (
            project_root / results_dir if results_dir is not None else project_root
        )
        selected_dir.mkdir(parents=True, exist_ok=True)
        if stats_text is not None:
            (selected_dir / "stats.json").write_text(stats_text, encoding="utf-8")
        if events_text is not None:
            (selected_dir / "events.json").write_text(events_text, encoding="utf-8")

        configured_dir = (
            ""
            if use_history
            else str(results_dir) if results_dir is not None else "."
        )
        with patch.dict(os.environ, {"TRAFFIC_RESULTS_DIR": configured_dir}):
            return AppTest.from_file(str(app_path)).run()

    @staticmethod
    def copy_traffic_analysis(project_root):
        package_dir = project_root / "traffic_analysis"
        package_dir.mkdir()
        for module in ("__init__.py", "analytics.py", "run_history.py"):
            shutil.copyfile(
                SOURCE_PROJECT / "traffic_analysis" / module,
                package_dir / module,
            )

    def create_run(
        self, project_root, run_id, completed_at, stats, events, source_video
    ):
        outputs_dir = project_root / "outputs"
        run_dir = outputs_dir / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
        (run_dir / "events.json").write_text(events, encoding="utf-8")
        register_completed_run(
            outputs_dir,
            run_id=run_id,
            run_directory=run_dir,
            completed_at=completed_at,
            source_video=source_video,
        )
        return run_dir

    @staticmethod
    def make_events(crossings_by_class=None, wrong_way=0, helmet_violations=0):
        crossings_by_class = crossings_by_class or {}
        events = []
        event_number = 0
        for vehicle_class, count in crossings_by_class.items():
            for _ in range(count):
                event_number += 1
                events.append({
                    "event_id": f"session-{event_number}",
                    "session_id": "session",
                    "event_type": "vehicle_crossing",
                    "timestamp": "2026-10-19T17:00:00Z",
                    "track_id": event_number,
                    "vehicle_class": vehicle_class,
                    "direction": "UP",
                    "counting_line": "main_line",
                })
        for event_type, count, vehicle_class in (
            ("wrong_way", wrong_way, "Car"),
            ("helmet_violation", helmet_violations, "Bike"),
        ):
            for _ in range(count):
                event_number += 1
                events.append({
                    "event_id": f"session-{event_number}",
                    "session_id": "session",
                    "event_type": event_type,
                    "timestamp": "2026-10-19T17:00:00Z",
                    "track_id": event_number,
                    "vehicle_class": vehicle_class,
                    "direction": "UP",
                    "counting_line": None,
                })
        return json.dumps(events)

    def test_valid_stats_render_both_download_buttons(self):
        stats = {
            "total": 12,
            "Car": 8,
            "Bike": 4,
            "Bus": 0,
            "Truck": 0,
            "unique_vehicle_total": 9,
            "unique_vehicles_by_class": {
                "Car": 5,
                "Bike": 4,
                "Bus": 0,
                "Truck": 0,
            },
            "line_crossings_total": 12,
            "line_crossings_by_class": {
                "Car": 8,
                "Bike": 4,
                "Bus": 0,
                "Truck": 0,
            },
            "untracked_detections": 3,
            "density": "MEDIUM",
            "wrong_way": 2,
        }

        app = self.run_dashboard(
            json.dumps(stats),
            self.make_events({"Car": 8, "Bike": 4}, wrong_way=2),
        )

        self.assertFalse(app.exception)
        buttons = app.get("download_button")
        self.assertEqual(len(buttons), 2)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Estimated Unique Vehicles"], "9")
        self.assertEqual(metrics["Car (unique)"], "5")
        self.assertEqual(metrics["Line Crossings"], "12")
        self.assertEqual(metrics["Bike crossings"], "4")
        self.assertEqual(metrics["Untracked Detection Observations"], "3")
        self.assertEqual(metrics["Wrong-Way Events"], "2")
        self.assertEqual(metrics["Helmet Violations"], "0")

    def test_null_unique_metrics_render_unavailable_not_zero(self):
        stats = {
            "total": 12,
            "unique_vehicle_count": None,
            "unique_vehicle_total": None,
            "unique_vehicles_by_class": None,
            "line_crossings_total": 12,
            "line_crossings_by_class": {
                "Car": 8,
                "Bike": 4,
                "Bus": 0,
                "Truck": 0,
            },
        }

        app = self.run_dashboard(
            json.dumps(stats),
            self.make_events({"Car": 2}),
        )

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Estimated Unique Vehicles"], "Unavailable")
        self.assertEqual(metrics["Car (unique)"], "Unavailable")
        self.assertEqual(metrics["Line Crossings"], "2")

    def test_missing_stats_show_exports_unavailable(self):
        app = self.run_dashboard()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertTrue(
            any("Exports are unavailable" in element.value for element in app.info)
        )

    def test_invalid_stats_show_error_without_crashing(self):
        app = self.run_dashboard("[]")

        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertTrue(
            any("Could not read traffic statistics" in element.value for element in app.error)
        )

    def test_alternate_results_directory_drives_all_displayed_run_metrics(self):
        root_stats = {
            "unique_vehicle_total": 100,
            "unique_vehicles_by_class": {"Car": 100},
            "density": "LOW",
        }
        selected_stats = {
            "unique_vehicle_total": 78,
            "unique_vehicles_by_class": {
                "Car": 24,
                "Bike": 36,
                "Bus": 1,
                "Truck": 17,
            },
            "line_crossings_total": 14,
            "line_crossings_by_class": {
                "Car": 2,
                "Bike": 10,
                "Bus": 0,
                "Truck": 2,
            },
            "untracked_detections": 0,
            "density": "MEDIUM",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "dashboard").mkdir()
            app_path = root / "dashboard" / "app.py"
            app_path.write_text(
                SOURCE_APP.read_text(encoding="utf-8-sig"), encoding="utf-8"
            )
            (root / "stats.json").write_text(
                json.dumps(root_stats), encoding="utf-8"
            )
            (root / "events.json").write_text(
                self.make_events({"Car": 1}, wrong_way=4), encoding="utf-8"
            )
            results_dir = root / "live" / "outputs"
            results_dir.mkdir(parents=True)
            (results_dir / "stats.json").write_text(
                json.dumps(selected_stats), encoding="utf-8"
            )
            (results_dir / "events.json").write_text(
                self.make_events(
                    {"Car": 2, "Bike": 10, "Truck": 2},
                    wrong_way=1,
                    helmet_violations=9,
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TRAFFIC_RESULTS_DIR": "live/outputs"}):
                app = AppTest.from_file(str(app_path)).run()

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Estimated Unique Vehicles"], "78")
        self.assertEqual(metrics["Car (unique)"], "24")
        self.assertEqual(metrics["Bike (unique)"], "36")
        self.assertEqual(metrics["Line Crossings"], "14")
        self.assertEqual(metrics["Car crossings"], "2")
        self.assertEqual(metrics["Bike crossings"], "10")
        self.assertEqual(metrics["Truck crossings"], "2")
        self.assertEqual(metrics["Wrong-Way Events"], "1")
        self.assertEqual(metrics["Helmet Violations"], "9")
        self.assertEqual(metrics["Untracked Detection Observations"], "0")
        self.assertTrue(
            any("Current traffic density: MEDIUM" in element.value for element in app.info)
        )
        chart = next(
            element for element in app._tree
            if type(element).__name__ == "UnknownElement"
        )
        chart_data = arrow_ipc.open_stream(
            chart.proto.datasets[0].data.data
        ).read_all().to_pylist()
        chart_counts = {
            row["index -- streamlit-generated"]: row["value"]
            for row in chart_data
        }
        self.assertEqual(chart_counts, {
            "Car": 2,
            "Bike": 10,
            "Bus": 0,
            "Truck": 2,
        })
        self.assertEqual(
            int(metrics["Line Crossings"]),
            sum(chart_counts.values()),
        )
        self.assertTrue(
            any("live/outputs" in element.value for element in app.caption)
        )

    def test_latest_run_is_default_and_previous_run_selection_is_consistent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "dashboard").mkdir()
            app_path = root / "dashboard" / "app.py"
            app_path.write_text(
                SOURCE_APP.read_text(encoding="utf-8-sig"), encoding="utf-8"
            )
            self.copy_traffic_analysis(root)

            old_stats = {
                "unique_vehicle_total": 7,
                "unique_vehicles_by_class": {"Car": 7},
                "density": "LOW",
                "untracked_detections": 1,
            }
            old_events = self.make_events({"Car": 3}, wrong_way=2)
            self.create_run(
                root,
                "older-run",
                "2026-10-18T10:00:00Z",
                old_stats,
                old_events,
                "old_video.mp4",
            )
            latest_stats = {
                "unique_vehicle_total": 20,
                "unique_vehicles_by_class": {"Bike": 20},
                "density": "HIGH",
                "untracked_detections": 4,
            }
            latest_events = self.make_events(
                {"Bike": 5}, wrong_way=1, helmet_violations=3
            )
            self.create_run(
                root,
                "latest-run",
                "2026-10-19T10:00:00Z",
                latest_stats,
                latest_events,
                "latest_video.mp4",
            )

            with patch.dict(os.environ, {"TRAFFIC_RESULTS_DIR": ""}):
                app = AppTest.from_file(str(app_path)).run()

            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(metrics["Estimated Unique Vehicles"], "20")
            self.assertEqual(metrics["Bike (unique)"], "20")
            self.assertEqual(metrics["Line Crossings"], "5")
            self.assertEqual(metrics["Bike crossings"], "5")
            self.assertEqual(metrics["Car crossings"], "0")
            self.assertEqual(metrics["Wrong-Way Events"], "1")
            self.assertEqual(metrics["Helmet Violations"], "3")
            self.assertTrue(
                any("latest_video.mp4" in element.value for element in app.caption)
            )
            self.assertTrue(
                any("2026-10-19" in element.value for element in app.caption)
            )

            app.selectbox[0].set_value("older-run").run()
            previous_metrics = {
                metric.label: metric.value for metric in app.metric
            }
            self.assertEqual(previous_metrics["Estimated Unique Vehicles"], "7")
            self.assertEqual(previous_metrics["Car (unique)"], "7")
            self.assertEqual(previous_metrics["Line Crossings"], "3")
            self.assertEqual(previous_metrics["Car crossings"], "3")
            self.assertEqual(previous_metrics["Bike crossings"], "0")
            self.assertEqual(previous_metrics["Wrong-Way Events"], "2")
            self.assertEqual(previous_metrics["Helmet Violations"], "0")
            self.assertTrue(
                any("old_video.mp4" in element.value for element in app.caption)
            )
            self.assertTrue(
                any("2026-10-18" in element.value for element in app.caption)
            )
            chart = next(
                element for element in app._tree
                if type(element).__name__ == "UnknownElement"
            )
            chart_data = arrow_ipc.open_stream(
                chart.proto.datasets[0].data.data
            ).read_all().to_pylist()
            chart_counts = {
                row["index -- streamlit-generated"]: row["value"]
                for row in chart_data
            }
            self.assertEqual(
                chart_counts,
                {"Car": 3, "Bike": 0, "Bus": 0, "Truck": 0},
            )
            self.assertEqual(
                sum(chart_counts.values()),
                int(previous_metrics["Line Crossings"]),
            )

    def test_default_dashboard_does_not_treat_unregistered_files_as_a_run(self):
        app = self.run_dashboard(
            json.dumps({"unique_vehicle_total": 100, "density": "LOW"}),
            self.make_events({"Car": 4}),
            use_history=True,
        )

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Estimated Unique Vehicles"], "Unavailable")
        self.assertEqual(metrics["Line Crossings"], "Unavailable")
        self.assertFalse(app.selectbox)

    def test_missing_event_file_does_not_fall_back_to_stats_crossings(self):
        app = self.run_dashboard(json.dumps({
            "total": 12,
            "line_crossings_total": 12,
            "line_crossings_by_class": {"Car": 12},
        }))

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Line Crossings"], "Unavailable")
        self.assertEqual(metrics["Car crossings"], "Unavailable")
        self.assertEqual(metrics["Wrong-Way Events"], "Unavailable")

    def test_null_and_missing_stats_metrics_are_unavailable(self):
        stats = {
            "unique_vehicle_total": None,
            "unique_vehicles_by_class": None,
            "untracked_detections": None,
            "density": None,
        }
        app = self.run_dashboard(
            json.dumps(stats),
            self.make_events(),
        )

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Estimated Unique Vehicles"], "Unavailable")
        self.assertEqual(metrics["Car (unique)"], "Unavailable")
        self.assertEqual(metrics["Untracked Detection Observations"], "Unavailable")
        self.assertEqual(metrics["Line Crossings"], "0")
        self.assertTrue(
            any("Current traffic density: Unavailable" in element.value for element in app.info)
        )

    def test_streamlit_script_directory_does_not_shadow_app_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            dashboard_dir = project_root / "dashboard"
            app_package = project_root / "app"
            analytics_package = app_package / "analytics"
            dashboard_dir.mkdir()
            analytics_package.mkdir(parents=True)

            shutil.copyfile(SOURCE_APP, dashboard_dir / "app.py")
            shutil.copyfile(SOURCE_PROJECT / "app" / "__init__.py", app_package / "__init__.py")
            shutil.copyfile(
                SOURCE_PROJECT / "app" / "analytics" / "__init__.py",
                analytics_package / "__init__.py",
            )
            shutil.copyfile(
                SOURCE_PROJECT / "app" / "analytics" / "exporter.py",
                analytics_package / "exporter.py",
            )
            traffic_analysis_package = project_root / "traffic_analysis"
            traffic_analysis_package.mkdir()
            for module in ("__init__.py", "analytics.py", "run_history.py"):
                shutil.copyfile(
                    SOURCE_PROJECT / "traffic_analysis" / module,
                    traffic_analysis_package / module,
                )
            (project_root / "stats.json").write_text(
                json.dumps({"total": 1, "Car": 1}), encoding="utf-8"
            )

            script = """
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

dashboard_dir = Path(sys.argv[1])
sys.path.insert(0, str(dashboard_dir))
import os
os.environ["TRAFFIC_RESULTS_DIR"] = "."
app = AppTest.from_file(str(dashboard_dir / "app.py")).run()
if app.exception:
    raise AssertionError(str(app.exception[0].message))
assert len(app.get("download_button")) == 2
"""
            result = subprocess.run(
                [sys.executable, "-c", script, str(dashboard_dir)],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
