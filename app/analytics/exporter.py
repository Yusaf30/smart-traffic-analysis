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

    rows = [
        ("metric", "value"),
        ("total", stats.get("total", 0)),
        ("by_class", json.dumps(stats.get("by_class", {}), sort_keys=True)),
        ("by_direction", json.dumps(stats.get("by_direction", {}), sort_keys=True)),
        ("density", stats.get("density", "UNKNOWN")),
        ("wrong_way", stats.get("wrong_way", 0)),
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
