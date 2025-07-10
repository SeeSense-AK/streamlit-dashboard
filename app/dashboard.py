import streamlit as st
import pandas as pd
from data import load_data
from ml_models import load_xgb_model, predict_risk
from visualisation import (
    create_hotspot_map,
    create_route_map,
    analyze_time_series,
    detect_anomalies,
    create_correlation_heatmap,
)
from recommendations import generate_safety_recommendations

# Set page config
st.set_page_config(
    page_title="Cycling Safety Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚴 Cycling Safety Analytics Dashboard")

# Sidebar: Data selection
st.sidebar.header("Data Options")
data_option = st.sidebar.selectbox(
    "Select Data",
    ("Sample Data", "Upload CSV")
)

data = load_data()

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Hotspot Map",
    "Route Map",
    "Time Series",
    "Anomaly Detection",
    "Recommendations"
])

with tab1:
    st.header("Braking & Swerving Hotspots")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Braking Hotspots")
        braking_map = create_hotspot_map(
            data[data["event_type"] == "braking"], 
            color_scale="OrRd",
            title="Braking Hotspots"
        )
        st.plotly_chart(braking_map, use_container_width=True)
    with col2:
        st.subheader("Swerving Hotspots")
        swerving_map = create_hotspot_map(
            data[data["event_type"] == "swerving"], 
            color_scale="YlGnBu",
            title="Swerving Hotspots"
        )
        st.plotly_chart(swerving_map, use_container_width=True)

with tab2:
    st.header("Popular Cycling Routes")
    st.pydeck_chart(create_route_map(data[data["event_type"] == "route"]))

with tab3:
    st.header("Incident Time Series Analysis")
    date_column = st.selectbox("Select date/time column", [c for c in data.columns if "date" in c.lower()])
    incident_column = st.selectbox("Select incident count/metric", [c for c in data.columns if c not in ["lat", "lon", "route_id", "event_type"]])
    ts_fig = analyze_time_series(data[[date_column, incident_column]].rename(columns={date_column: "date", incident_column: "incidents"}))
    st.plotly_chart(ts_fig, use_container_width=True)

with tab4:
    st.header("Anomaly Detection")
    anomaly_metric = st.selectbox("Select metric for anomaly detection", [c for c in data.columns if c not in ["lat", "lon", "route_id", "event_type"]])
    anomalies_fig, anomalies = detect_anomalies(data.set_index(date_column), column=anomaly_metric)
    st.plotly_chart(anomalies_fig, use_container_width=True)
    st.subheader("Detected Anomalies")
    st.dataframe(anomalies)

with tab5:
    st.header("AI-Powered Safety Recommendations")
    with st.spinner("Generating recommendations..."):
        braking_data = data[data["event_type"] == "braking"]
        swerving_data = data[data["event_type"] == "swerving"]
        recommendations_df = generate_safety_recommendations(braking_data, swerving_data)
    st.dataframe(recommendations_df)

    st.markdown("### Correlation Analysis")
    corr_fig = create_correlation_heatmap(data)
    st.plotly_chart(corr_fig, use_container_width=True)

# Sidebar: Risk Prediction
st.sidebar.header("Risk Prediction")
if st.sidebar.checkbox("Enable Risk Prediction"):
    st.sidebar.markdown("Input features for risk prediction below:")
    features = {}
    model = load_xgb_model()
    # Example feature fields - adjust based on your model:
    features["distinct_cyclists"] = st.sidebar.number_input("Distinct Cyclists", min_value=0, value=100)
    features["popularity_rating"] = st.sidebar.slider("Popularity Rating", 0.0, 10.0, 5.0)
    features["avg_speed"] = st.sidebar.number_input("Average Speed (km/h)", min_value=0.0, value=15.0)
    features["has_bike_lane"] = st.sidebar.radio("Bike Lane Present?", ["Yes", "No"]) == "Yes"
    X_pred = pd.DataFrame([features])
    pred = predict_risk(model, X_pred)
    st.sidebar.success(f"Predicted Risk Level: {pred[0]}")
