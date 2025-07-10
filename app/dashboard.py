import streamlit as st
import pandas as pd
import numpy as np
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
from datetime import datetime, timedelta

# Custom CSS from db_test.py
def load_css():
    st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #ffffff; border-radius: 4px 4px 0px 0px;
        padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #4e89ae; color: white; }
    .metric-card {
        background-color: #ffffff; border-radius: 5px; padding: 15px;
        border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .card-title { color: #555; font-size: 14px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
    .card-value { color: #333; font-size: 24px; font-weight: 700; }
    .card-delta { color: #28a745; font-size: 14px; font-weight: 500; }
    .card-delta-down { color: #dc3545; }
    .section-title {
        font-size: 20px; font-weight: 600; color: #333;
        margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eaeaea;
    }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    .widget-label { font-weight: 600; margin-bottom: 5px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

def metric_card(title, value, delta=None, delta_sign=None, suffix=""):
    delta_class = "card-delta-down" if delta_sign == "negative" else ""
    delta_html = f'<div class="card-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}{suffix}</div>
        {delta_html}
    </div>
    """

# Set page config
st.set_page_config(
    page_title="Cycling Safety Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()
st.title("🚴 Cycling Safety Analytics Dashboard")

# Sidebar filters from db_test.py
st.sidebar.markdown('<div class="widget-label">Data Filters</div>', unsafe_allow_html=True)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now())
)
min_popularity = st.sidebar.slider("Min. Route Popularity", 1, 10, 1)
min_cyclists = st.sidebar.slider("Min. Distinct Cyclists", 10, 1000, 50)

# Load data
data_dict = load_data()
braking_data = data_dict['braking_data']
swerving_data = data_dict['swerving_data']
route_data = data_dict['route_data']
time_series_data = data_dict['time_series_data']

# Apply filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    time_series_data = time_series_data[(time_series_data['date'] >= start_date) & (time_series_data['date'] <= end_date)]
    braking_data = braking_data[(braking_data['date_recorded'] >= start_date) & (braking_data['date_recorded'] <= end_date)]
    swerving_data = swerving_data[(swerving_data['date_recorded'] >= start_date) & (swerving_data['date_recorded'] <= end_date)]
filtered_routes = route_data[
    (route_data['popularity_rating'] >= min_popularity) &
    (route_data['distinct_cyclists'] >= min_cyclists)
]

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Hotspot Map", "Route Map", "Time Series", "Anomaly Detection", "Recommendations"
])

with tab1:
    st.markdown('<div class="section-title">Braking & Swerving Hotspots</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(metric_card("Braking Hotspots", len(braking_data)), unsafe_allow_html=True)
        braking_map = create_hotspot_map(braking_data, color_scale="OrRd", title="Braking Hotspots")
        st.plotly_chart(braking_map, use_container_width=True)
    with col2:
        st.markdown(metric_card("Swerving Hotspots", len(swerving_data)), unsafe_allow_html=True)
        swerving_map = create_hotspot_map(swerving_data, color_scale="YlGnBu", title="Swerving Hotspots")
        st.plotly_chart(swerving_map, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Popular Cycling Routes</div>', unsafe_allow_html=True)
    st.markdown(metric_card("Total Routes Analyzed", len(filtered_routes)), unsafe_allow_html=True)
    sample_size = st.slider("Number of Routes to Display", 100, 2000, 1000)
    filtered_routes_sample = filtered_routes.sample(min(sample_size, len(filtered_routes))) if not filtered_routes.empty else pd.DataFrame()
    if filtered_routes_sample.empty:
        st.warning("No routes match the selected criteria.")
    else:
        st.pydeck_chart(create_route_map(filtered_routes_sample))

with tab3:
    st.markdown('<div class="section-title">Incident Time Series Analysis</div>', unsafe_allow_html=True)
    incident_column = st.selectbox("Select metric", ['incidents', 'avg_braking_events', 'avg_swerving_events'])
    ts_fig = analyze_time_series(time_series_data[['date', incident_column]].rename(columns={incident_column: "incidents"}))
    st.plotly_chart(ts_fig, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)
    anomaly_metric = st.selectbox("Select metric for anomaly detection", ['incidents', 'avg_braking_events', 'avg_swerving_events'])
    anomalies_fig, anomalies = detect_anomalies(time_series_data.set_index('date'), column=anomaly_metric)
    st.plotly_chart(anomalies_fig, use_container_width=True)
    st.subheader("Detected Anomalies")
    st.dataframe(anomalies.reset_index()[['date', anomaly_metric, 'anomaly_score']])

with tab5:
    st.markdown('<div class="section-title">AI-Powered Safety Recommendations</div>', unsafe_allow_html=True)
    recommendations_df = generate_safety_recommendations(braking_data, swerving_data)
    st.markdown(metric_card("High Priority Actions", len(recommendations_df[recommendations_df['Priority'] == 'High'])), unsafe_allow_html=True)
    st.dataframe(recommendations_df)
    st.markdown("### Correlation Analysis")
    combined_data = pd.concat([braking_data, swerving_data, route_data, time_series_data], ignore_index=True)
    corr_fig = create_correlation_heatmap(combined_data)
    st.plotly_chart(corr_fig, use_container_width=True)

# Sidebar: Risk Prediction
st.sidebar.header("Risk Prediction")
if st.sidebar.checkbox("Enable Risk Prediction"):
    features = {}
    model = load_xgb_model()
    features["distinct_cyclists"] = st.sidebar.number_input("Distinct Cyclists", min_value=0, value=100)
    features["popularity_rating"] = st.sidebar.slider("Popularity Rating", 0.0, 10.0, 5.0)
    features["avg_speed"] = st.sidebar.number_input("Average Speed (km/h)", min_value=0.0, value=15.0)
    features["has_bike_lane"] = st.sidebar.radio("Bike Lane Present?", ["Yes", "No"]) == "Yes"
    X_pred = pd.DataFrame([features])
    pred = predict_risk(model, X_pred)
    st.sidebar.success(f"Predicted Risk Level: {pred[0]}")
