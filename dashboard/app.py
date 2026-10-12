import json
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
STATS_FILE = PROJECT_ROOT / "stats.json"

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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Vehicles", stats.get("total", 0))
        col2.metric("Cars", stats.get("Car", 0))
        col3.metric("Bikes", stats.get("Bike", 0))
        col4.metric("Wrong-Way Events", stats.get("wrong_way", 0))

        st.subheader("Traffic Density")
        st.info(f"Current traffic density: {stats.get('density', 'Unknown')}")

        col5, col6 = st.columns(2)
        col5.metric("Buses", stats.get("Bus", 0))
        col6.metric("Trucks", stats.get("Truck", 0))

    except (json.JSONDecodeError, OSError) as error:
        st.error(f"Could not read traffic statistics: {error}")

st.subheader("Exports")
st.info("CSV and JSON export integration will be added in a later step.")
