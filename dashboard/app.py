import streamlit as st

st.set_page_config(
    page_title="Traffic Analysis Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Analysis Dashboard")
st.caption("Traffic monitoring and analytics")

st.subheader("Traffic Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Vehicles", "—")
col2.metric("Cars", "—")
col3.metric("Bikes", "—")
col4.metric("Wrong-Way Events", "—")

st.subheader("Traffic Density")
st.info("Traffic statistics will appear here when analytics data is connected.")

st.subheader("Exports")
st.write("CSV and JSON export integration will be added in a later step.")
