import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


SOURCE_APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
SOURCE_PROJECT = SOURCE_APP.parents[1]


class TestDashboardExports(unittest.TestCase):
    def run_dashboard(self, stats_text=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        project_root = Path(temp_dir.name)
        dashboard_dir = project_root / "dashboard"
        dashboard_dir.mkdir()
        app_path = dashboard_dir / "app.py"
        app_path.write_text(SOURCE_APP.read_text(encoding="utf-8-sig"), encoding="utf-8")
        if stats_text is not None:
            (project_root / "stats.json").write_text(stats_text, encoding="utf-8")

        return AppTest.from_file(str(app_path)).run()

    def test_valid_stats_render_both_download_buttons(self):
        stats = {
            "total": 12,
            "Car": 8,
            "Bike": 4,
            "Bus": 0,
            "Truck": 0,
            "density": "MEDIUM",
            "wrong_way": 2,
        }

        app = self.run_dashboard(json.dumps(stats))

        self.assertFalse(app.exception)
        buttons = app.get("download_button")
        self.assertEqual(len(buttons), 2)
        self.assertEqual(app.metric[0].value, "12")

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
            (project_root / "stats.json").write_text(
                json.dumps({"total": 1, "Car": 1}), encoding="utf-8"
            )

            script = """
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

dashboard_dir = Path(sys.argv[1])
sys.path.insert(0, str(dashboard_dir))
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
