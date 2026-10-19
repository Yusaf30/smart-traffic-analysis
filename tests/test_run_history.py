import json

import pytest

from traffic_analysis.run_history import (
    get_run_history,
    latest_completed_run,
    list_completed_runs,
    register_completed_run,
)


def _make_run(outputs_dir, run_id):
    run_dir = outputs_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "stats.json").write_text("{}", encoding="utf-8")
    (run_dir / "events.json").write_text("[]", encoding="utf-8")
    return run_dir


def _register(outputs_dir, run_id, completed_at):
    run_dir = _make_run(outputs_dir, run_id)
    register_completed_run(
        outputs_dir,
        run_id=run_id,
        run_directory=run_dir,
        completed_at=completed_at,
        source_video="/video/source.mp4",
    )


def test_latest_completed_run_advances_without_overwriting_previous_runs(tmp_path):
    outputs_dir = tmp_path / "outputs"
    _register(outputs_dir, "older-run", "2026-10-18T10:00:00Z")
    _register(outputs_dir, "newer-run", "2026-10-19T10:00:00Z")

    manifest = json.loads(
        (outputs_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    runs, latest = get_run_history(outputs_dir)

    assert manifest["latest_run_id"] == "newer-run"
    assert [run["run_id"] for run in runs] == ["newer-run", "older-run"]
    assert latest == runs[0]
    assert latest_completed_run(outputs_dir)["run_id"] == "newer-run"
    assert (outputs_dir / "runs" / "older-run" / "stats.json").is_file()
    assert list_completed_runs(outputs_dir) == runs


def test_incomplete_run_cannot_replace_last_completed_run(tmp_path):
    outputs_dir = tmp_path / "outputs"
    _register(outputs_dir, "successful-run", "2026-10-18T10:00:00Z")
    incomplete_dir = _make_run(outputs_dir, "incomplete-run")
    (incomplete_dir / "events.json").unlink()

    with pytest.raises(ValueError, match="missing events.json"):
        register_completed_run(
            outputs_dir,
            run_id="incomplete-run",
            run_directory=incomplete_dir,
            completed_at="2026-10-19T10:00:00Z",
            source_video="failed.mp4",
        )

    manifest = json.loads(
        (outputs_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["latest_run_id"] == "successful-run"
    assert [run["run_id"] for run in list_completed_runs(outputs_dir)] == [
        "successful-run"
    ]


def test_manifest_latest_pointer_falls_back_if_pointed_run_is_incomplete(tmp_path):
    outputs_dir = tmp_path / "outputs"
    _register(outputs_dir, "completed-run", "2026-10-18T10:00:00Z")
    incomplete_dir = outputs_dir / "runs" / "partial-run"
    incomplete_dir.mkdir()
    (incomplete_dir / "stats.json").write_text("{}", encoding="utf-8")
    manifest_path = outputs_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"].append({
        "run_id": "partial-run",
        "directory": "runs/partial-run",
        "completed_at": "2026-10-19T10:00:00Z",
        "source_video": "partial.mp4",
    })
    manifest["latest_run_id"] = "partial-run"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    runs, latest = get_run_history(outputs_dir)

    assert [run["run_id"] for run in runs] == ["completed-run"]
    assert latest["run_id"] == "completed-run"


def test_manifest_is_created_only_after_required_artifacts_exist(tmp_path):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "runs" / "partial"
    run_dir.mkdir(parents=True)
    (run_dir / "stats.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing events.json"):
        register_completed_run(
            outputs_dir,
            run_id="partial",
            run_directory=run_dir,
            completed_at="2026-10-19T10:00:00Z",
            source_video="partial.mp4",
        )

    assert not (outputs_dir / "run_manifest.json").exists()
