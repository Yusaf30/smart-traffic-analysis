import json
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
STATS_FILE = PROJECT_ROOT / "stats.json"
stats = None

st.subheader("Traffic Overview")

if not STATS_FILE.exists():
    st.warning(
        "Traffic statistics are not available yet. "
        "Run the computer-vision script with a valid input video "
        "to generate stats.json."
    )
else:
    try:
        with STATS_FILE.open("r", encoding="utf-8") as file:
            stats = json.load(file)
        if not isinstance(stats, dict):
            raise ValueError("Statistics must contain a JSON object.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Line Crossings", stats.get("total", 0))
        col2.metric("Cars", stats.get("Car", 0))
        col3.metric("Bikes", stats.get("Bike", 0))
        col4.metric("Wrong-Way Events", stats.get("wrong_way", 0))

        st.subheader("Traffic Density")
        st.info(f"Current traffic density: {stats.get('density', 'Unknown')}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Buses", stats.get("Bus", 0))
        col6.metric("Trucks", stats.get("Truck", 0))
        col7.metric(
            "Estimated Unique Vehicles",
            stats.get("unique_vehicle_count", "Unavailable"),
        )
        col8.metric(
            "Raw Tracker IDs",
            stats.get("raw_tracker_ids_observed", "Unavailable"),
        )

        st.caption(
            f"Frame detections observed: "
            f"{stats.get('total_frame_detections', 'Unavailable')}"
        )

    except (json.JSONDecodeError, OSError, ValueError) as error:
        st.error(f"Could not read traffic statistics: {error}")
        stats = None

st.subheader("Event Analytics")

EVENTS_FILE = PROJECT_ROOT / "events.json"

if not EVENTS_FILE.exists():
    st.info(
        "Event data is not available yet. "
        "Run the computer-vision pipeline to generate events.json."
    )
else:
    try:
        from traffic_analysis.analytics import load_events, summarize_events

        events = load_events(EVENTS_FILE)
        summary = summarize_events(events)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Crossings", summary["total_crossings"])
        col2.metric("Wrong-Way Events", summary["wrong_way_events"])
        col3.metric("Helmet Violations", summary["helmet_violations"])

        st.write("Vehicle Crossings by Class")
        st.bar_chart(summary["vehicle_counts"])

        st.caption(f"Recorded sessions: {len(summary['session_ids'])}")

    except (ValueError, OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read event analytics: {error}")

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
