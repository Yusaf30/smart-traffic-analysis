import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Traffic Analysis Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Analysis Dashboard")
st.caption("Traffic monitoring and analytics")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not sys.path or sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))
from traffic_analysis.run_history import format_run_label, get_run_history

configured_results_dir = os.environ.get("TRAFFIC_RESULTS_DIR", "").strip()
selected_run = None
if configured_results_dir:
    RESULTS_DIR = Path(configured_results_dir).expanduser()
    if not RESULTS_DIR.is_absolute():
        RESULTS_DIR = PROJECT_ROOT / RESULTS_DIR
    RESULTS_DIR = RESULTS_DIR.resolve()
    st.caption(f"Using explicit results directory: {RESULTS_DIR}")
else:
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    try:
        completed_runs, latest_run = get_run_history(OUTPUTS_DIR)
        if completed_runs:
            run_by_id = {run["run_id"]: run for run in completed_runs}
            run_ids = [run["run_id"] for run in completed_runs]
            latest_index = (
                run_ids.index(latest_run["run_id"])
                if latest_run is not None
                else 0
            )
            selected_run_id = st.selectbox(
                "Processing run",
                options=run_ids,
                index=latest_index,
                format_func=lambda run_id: format_run_label(run_by_id[run_id]),
            )
            selected_run = run_by_id[selected_run_id]
            RESULTS_DIR = OUTPUTS_DIR / selected_run["directory"]
            st.caption(f"Selected run: {format_run_label(selected_run)}")
        else:
            RESULTS_DIR = None
            st.info(
                "No completed runs are recorded yet. "
                "Completed runs will appear under outputs/runs."
            )
    except (OSError, ValueError) as error:
        RESULTS_DIR = None
        st.error(f"Could not read run history: {error}")

STATS_FILE = RESULTS_DIR / "stats.json" if RESULTS_DIR else None
EVENTS_FILE = RESULTS_DIR / "events.json" if RESULTS_DIR else None
stats = None
event_summary = None
stats_error = None
events_error = None


def display_metric_value(value):
    return "Unavailable" if value is None else value


if STATS_FILE is not None and STATS_FILE.exists():
    try:
        with STATS_FILE.open("r", encoding="utf-8") as file:
            stats = json.load(file)
        if not isinstance(stats, dict):
            raise ValueError("Statistics must contain a JSON object.")
    except (json.JSONDecodeError, OSError, ValueError) as error:
        stats_error = error
        stats = None

if EVENTS_FILE is not None and EVENTS_FILE.exists():
    try:
        from traffic_analysis.analytics import load_events, summarize_events

        event_summary = summarize_events(load_events(EVENTS_FILE))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        events_error = error

st.subheader("Traffic Overview")

if stats is None:
    if stats_error:
        st.error(f"Could not read traffic statistics: {stats_error}")
    elif STATS_FILE is not None:
        st.warning(f"Traffic statistics are not available in {STATS_FILE}.")

unique_total = (
    stats.get("unique_vehicle_total", stats.get("unique_vehicle_count"))
    if stats is not None
    else None
)
unique_by_class = (
    stats.get("unique_vehicles_by_class") if stats is not None else None
)
event_metrics = event_summary or {}
overview_columns = st.columns(5)
overview_columns[0].metric(
    "Estimated Unique Vehicles", display_metric_value(unique_total)
)
overview_columns[1].metric(
    "Line Crossings",
    display_metric_value(event_metrics.get("total_crossings")),
)
overview_columns[2].metric(
    "Untracked Detection Observations",
    display_metric_value(
        stats.get("untracked_detections") if stats is not None else None
    ),
)
overview_columns[3].metric(
    "Wrong-Way Events",
    display_metric_value(event_metrics.get("wrong_way_events")),
)
overview_columns[4].metric(
    "Helmet Violations",
    display_metric_value(event_metrics.get("helmet_violations")),
)

st.write("Estimated Unique Vehicles by Class")
unique_columns = st.columns(4)
for column, vehicle_class in zip(
    unique_columns, ("Car", "Bike", "Bus", "Truck")
):
    value = (
        unique_by_class.get(vehicle_class)
        if isinstance(unique_by_class, dict)
        else None
    )
    column.metric(f"{vehicle_class} (unique)", display_metric_value(value))

st.write("Line Crossings by Class")
crossings_by_class = event_metrics.get("vehicle_counts")
crossing_columns = st.columns(4)
for column, vehicle_class in zip(
    crossing_columns, ("Car", "Bike", "Bus", "Truck")
):
    value = (
        crossings_by_class.get(vehicle_class)
        if isinstance(crossings_by_class, dict)
        else None
    )
    column.metric(
        f"{vehicle_class} crossings",
        display_metric_value(value),
    )

st.subheader("Traffic Density")
st.info(
    "Current traffic density: "
    + display_metric_value(stats.get("density") if stats is not None else None)
)

st.caption(
    f"Frame detections observed: "
    f"{display_metric_value(stats.get('total_frame_detections') if stats else None)}"
)
st.caption(
    f"Raw tracker IDs observed: "
    f"{display_metric_value(stats.get('raw_tracker_ids_observed') if stats else None)}"
)

st.subheader("Event Analytics")

if event_summary is None:
    if events_error:
        st.error(f"Could not read event analytics: {events_error}")
    elif EVENTS_FILE is not None:
        st.info(f"Event data is not available in {EVENTS_FILE}.")
else:
    st.write("Vehicle Crossings by Class")
    st.bar_chart(event_summary["vehicle_counts"])
    st.caption(f"Recorded sessions: {len(event_summary['session_ids'])}")

st.subheader("Exports")
if stats is None:
    st.info("Exports are unavailable until valid traffic statistics are available.")
else:
    try:
        from app.analytics.exporter import (
            export_stats_csv,
            export_stats_json,
            prepare_stats_for_export,
        )

        export_stats = prepare_stats_for_export(stats)
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "traffic_stats.csv"
            json_path = Path(temp_dir) / "traffic_stats.json"
            export_stats_csv(export_stats, csv_path)
            export_stats_json(export_stats, json_path)
            csv_data = csv_path.read_bytes()
            json_data = json_path.read_bytes()

        csv_col, json_col = st.columns(2)
        csv_col.download_button(
            "Download CSV",
            data=csv_data,
            file_name="traffic_stats.csv",
            mime="text/csv",
        )
        json_col.download_button(
            "Download JSON",
            data=json_data,
            file_name="traffic_stats.json",
            mime="application/json",
        )
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Could not prepare statistics exports: {error}")
