"""Manifest-backed catalog of completed traffic-analysis runs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0"


def _manifest_path(outputs_dir: str | Path) -> Path:
    return Path(outputs_dir) / MANIFEST_NAME


def _read_manifest(outputs_dir: str | Path) -> dict[str, Any]:
    path = _manifest_path(outputs_dir)
    if not path.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "latest_run_id": None,
            "runs": [],
        }

    with path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("runs"), list)
    ):
        raise ValueError("Invalid run-history manifest.")
    if manifest.get("latest_run_id") is not None and not isinstance(
        manifest["latest_run_id"], str
    ):
        raise ValueError("Invalid latest_run_id in run-history manifest.")
    return manifest


def _run_directory(outputs_dir: str | Path, record: dict[str, Any]) -> Path | None:
    run_id = record.get("run_id")
    directory = record.get("directory")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(directory, str)
        or Path(directory).parts != ("runs", run_id)
    ):
        return None
    return Path(outputs_dir) / directory


def get_run_history(
    outputs_dir: str | Path,
) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    """Return valid completed runs and the manifest's latest valid run."""
    manifest = _read_manifest(outputs_dir)
    completed_runs = []
    for record in manifest["runs"]:
        if not isinstance(record, dict):
            continue
        run_directory = _run_directory(outputs_dir, record)
        if run_directory is None:
            continue
        if not (run_directory / "stats.json").is_file():
            continue
        if not (run_directory / "events.json").is_file():
            continue

        run_id = record.get("run_id")
        completed_at = record.get("completed_at")
        if not isinstance(run_id, str) or not isinstance(completed_at, str):
            continue
        source_video = record.get("source_video")
        completed_runs.append({
            "run_id": run_id,
            "directory": str(record["directory"]),
            "completed_at": completed_at,
            "source_video": (
                source_video if isinstance(source_video, str) else ""
            ),
        })
    completed_runs = sorted(
        completed_runs,
        key=lambda record: (record["completed_at"], record["run_id"]),
        reverse=True,
    )
    latest_run_id = manifest.get("latest_run_id")
    for record in completed_runs:
        if record["run_id"] == latest_run_id:
            return completed_runs, record
    latest = completed_runs[0] if completed_runs else None
    return completed_runs, latest


def list_completed_runs(outputs_dir: str | Path) -> list[dict[str, str]]:
    """Return manifest-listed runs with both required JSON artifacts present."""
    return get_run_history(outputs_dir)[0]


def latest_completed_run(outputs_dir: str | Path) -> dict[str, str] | None:
    """Return the latest valid completed run, honoring the manifest pointer."""
    return get_run_history(outputs_dir)[1]


def format_run_label(record: dict[str, str]) -> str:
    """Build a readable selector label from completion time and video name."""
    completed_at = record["completed_at"]
    try:
        timestamp = datetime.fromisoformat(
            completed_at.replace("Z", "+00:00")
        ).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        timestamp = completed_at
    source_video = Path(record["source_video"]).name or "Unknown video"
    return f"{timestamp} | {source_video} | {record['run_id']}"


def register_completed_run(
    outputs_dir: str | Path,
    *,
    run_id: str,
    run_directory: str | Path,
    completed_at: str,
    source_video: str | Path,
) -> None:
    """Append a completed run and atomically advance the latest-run pointer."""
    root = Path(outputs_dir)
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("Invalid run_id.")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("completed_at must be a non-empty string.")
    expected_directory = (root / "runs" / run_id).resolve()
    actual_directory = Path(run_directory).resolve()
    if actual_directory != expected_directory:
        raise ValueError("Run directory must be outputs/runs/<run_id>.")
    if not (actual_directory / "stats.json").is_file():
        raise ValueError("Completed run is missing stats.json.")
    if not (actual_directory / "events.json").is_file():
        raise ValueError("Completed run is missing events.json.")

    root.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(root)
    if any(
        isinstance(record, dict) and record.get("run_id") == run_id
        for record in manifest["runs"]
    ):
        raise FileExistsError(f"Run already registered: {run_id}")

    manifest["runs"].append({
        "run_id": run_id,
        "directory": f"runs/{run_id}",
        "completed_at": completed_at,
        "source_video": Path(source_video).name,
    })
    manifest["latest_run_id"] = run_id

    path = _manifest_path(root)
    temporary_path = root / f".{MANIFEST_NAME}.{uuid4().hex}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
