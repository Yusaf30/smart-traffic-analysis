import csv
import json
from pathlib import Path
from typing import Any


VEHICLE_CLASSES = ("Car", "Bike", "Bus", "Truck")


def prepare_stats_for_export(stats: dict[str, Any]) -> dict[str, Any]:
    """Adapt dashboard statistics to the exporter's aggregate structure."""
    export_stats = dict(stats)
    export_stats["by_class"] = {
        vehicle_class: stats.get(vehicle_class, 0)
        for vehicle_class in VEHICLE_CLASSES
    }
    return export_stats


def export_stats_csv(stats: dict[str, Any], file_path: str | Path) -> None:
    """Export aggregate traffic statistics as a two-column CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def json_metric(name: str) -> str:
        value = stats.get(name)
        return json.dumps(value, sort_keys=True) if isinstance(value, dict) else ""

    def scalar_metric(name: str) -> Any:
        value = stats.get(name)
        return "" if value is None else value

    rows = [
        ("metric", "value"),
        ("total", stats.get("total", 0)),
        ("by_class", json.dumps(stats.get("by_class", {}), sort_keys=True)),
        ("by_direction", json.dumps(stats.get("by_direction", {}), sort_keys=True)),
        ("density", stats.get("density", "UNKNOWN")),
        ("wrong_way", stats.get("wrong_way", 0)),
        ("unique_vehicle_total", scalar_metric("unique_vehicle_total")),
        ("unique_vehicles_by_class", json_metric("unique_vehicles_by_class")),
        ("line_crossings_total", scalar_metric("line_crossings_total")),
        ("line_crossings_by_class", json_metric("line_crossings_by_class")),
        ("untracked_detections", scalar_metric("untracked_detections")),
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def export_stats_json(stats: dict[str, Any], file_path: str | Path) -> None:
    """Export aggregate traffic statistics as formatted JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, sort_keys=True)
        file.write("\n")
