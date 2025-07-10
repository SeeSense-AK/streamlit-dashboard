import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    .btn-custom {
        background-color: #4e89ae; color: white; padding: 10px 15px;
        border-radius: 4px; cursor: pointer; text-align: center;
        width: 100%; margin-top: 10px; font-weight: 500;
    }
    .btn-custom:hover { background-color: #3a6d8a; }
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

# Sidebar filters
st.sidebar.markdown('<div class="widget-label">Data Filters</div>', unsafe_allow_html=True)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now())
)
min_popularity = st.sidebar.slider("Min. Route Popularity", 1, 10, 1)
min_cyclists = st.sidebar.slider("Min. Distinct Cyclists", 10, 1000, 50)
st.sidebar.markdown('<div class="btn-custom">Apply Filters</div>', unsafe_allow_html=True)

# Load data
try:
    data_dict = load_data()
    braking_data = data_dict['braking_data']
    swerving_data = data_dict['swerving_data']
    route_data = data_dict['route_data']
    time_series_data = data_dict['time_series_data']
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

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
    "Dashboard Overview", "ML Insights", "Spatial Analysis", "Advanced Analytics", "Actionable Insights"
])

with tab1:
    st.markdown('<div class="section-title">Key Safety Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Total Routes Analyzed", f"{len(filtered_routes):,}", "+5% vs prev month", "positive"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Braking Hotspots", f"{len(braking_data)}", "-12% vs prev month", "positive"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Swerving Hotspots", f"{len(swerving_data)}", "-8% vs prev month", "positive"), unsafe_allow_html=True)
    with col4:
        safety_score = round(np.mean([braking_data['intensity'].mean(), swerving_data['intensity'].mean()]), 1)
        st.markdown(metric_card("Safety Score", f"{safety_score}", "+0.6 vs prev month", "positive", "/10"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Safety Hotspot Maps</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        braking_map = create_hotspot_map(braking_data, color_scale="OrRd", title="Braking Hotspots")
        st.plotly_chart(braking_map, use_container_width=True)
    with col2:
        swerving_map = create_hotspot_map(swerving_data, color_scale="YlGnBu", title="Swerving Hotspots")
        st.plotly_chart(swerving_map, use_container_width=True)

    st.markdown('<div class="section-title">Route Popularity Map</div>', unsafe_allow_html=True)
    sample_size = st.slider("Number of Routes to Display", 100, 2000, 1000)
    filtered_routes_sample = filtered_routes.sample(min(sample_size, len(filtered_routes))) if not filtered_routes.empty else pd.DataFrame()
    if filtered_routes_sample.empty:
        st.warning("No routes match the selected criteria.")
    else:
        st.pydeck_chart(create_route_map(filtered_routes_sample))

    st.markdown('<div class="section-title">Incident Trends (Last 30 Days)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for metric, color in [
        ('incidents', '#e41a1c'),
        ('avg_braking_events', '#377eb8'),
        ('avg_swerving_events', '#4daf4a')
    ]:
        fig.add_trace(go.Scatter(
            x=time_series_data['date'],
            y=time_series_data[metric],
            mode='lines',
            name=metric.replace('_', ' ').title(),
            line=dict(color=color, width=2)
        ))
    fig.update_layout(
        title="Safety Incidents and Events Trend",
        xaxis_title="Date",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Machine Learning Insights</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-title">Risk Prediction</div>', unsafe_allow_html=True)
        model = load_xgb_model()
        X_sample = filtered_routes.sample(min(10, len(filtered_routes)))[['distinct_cyclists', 'popularity_rating', 'avg_speed', 'has_bike_lane']].copy()
        X_sample['has_bike_lane'] = X_sample['has_bike_lane'].astype(int)
        predictions = predict_risk(model, X_sample)
        prediction_df = pd.DataFrame({
            'Route ID': filtered_routes.loc[X_sample.index, 'route_id'].values,
            'Distinct Cyclists': X_sample['distinct_cyclists'].values,
            'Popularity': X_sample['popularity_rating'].values,
            'Avg. Speed (km/h)': X_sample['avg_speed'].values,
            'Has Bike Lane': X_sample['has_bike_lane'].values,
            'Predicted Risk Level': predictions
        })
        def color_risk(val):
            if val >= 7:
                return 'background-color: #ffcccc'
            elif val >= 4:
                return 'background-color: #ffffcc'
            else:
                return 'background-color: #ccffcc'
        styled_predictions = prediction_df.style.applymap(color_risk, subset=['Predicted Risk Level'])
        st.dataframe(styled_predictions, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Behavioral Clusters</div>', unsafe_allow_html=True)
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        kmeans_data = filtered_routes[['distinct_cyclists', 'popularity_rating', 'avg_speed', 'avg_duration']].copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(kmeans_data)
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
        kmeans.fit(scaled_data)
        filtered_routes['cluster'] = kmeans.labels_
        cluster_names = {0: 'Commuters', 1: 'Exercise Riders', 2: 'Leisure Cyclists', 3: 'Mixed Usage'}
        filtered_routes['cluster_name'] = filtered_routes['cluster'].map(cluster_names)
        fig = px.scatter(
            filtered_routes.sample(min(500, len(filtered_routes))),
            x='avg_speed',
            y='avg_duration',
            color='cluster_name',
            size='distinct_cyclists',
            hover_data=['route_id', 'popularity_rating'],
            title="Cyclist Behavior Clusters",
            labels={'avg_speed': 'Average Speed (km/h)', 'avg_duration': 'Average Trip Duration (min)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Spatial Analysis</div>', unsafe_allow_html=True)
    heatmap_type = st.selectbox("Select Heatmap Type", ["Braking Incidents", "Swerving Incidents", "Combined Safety Score"])
    if heatmap_type == "Braking Incidents":
        fig = px.density_mapbox(
            braking_data, lat='lat', lon='lon', z='intensity', radius=20,
            center=dict(lat=braking_data['lat'].mean(), lon=braking_data['lon'].mean()),
            zoom=12, mapbox_style="carto-positron", title="Braking Incidents Density"
        )
    elif heatmap_type == "Swerving Incidents":
        fig = px.density_mapbox(
            swerving_data, lat='lat', lon='lon', z='intensity', radius=20,
            center=dict(lat=swerving_data['lat'].mean(), lon=swerving_data['lon'].mean()),
            zoom=12, mapbox_style="carto-positron", title="Swerving Incidents Density"
        )
    else:
        combined_data = pd.concat([braking_data[['lat', 'lon', 'intensity']], swerving_data[['lat', 'lon', 'intensity']]])
        fig = px.density_mapbox(
            combined_data, lat='lat', lon='lon', z='intensity', radius=20,
            center=dict(lat=combined_data['lat'].mean(), lon=combined_data['lon'].mean()),
            zoom=12, mapbox_style="carto-positron", title="Combined Safety Score Density"
        )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Advanced Analytics</div>', unsafe_allow_html=True)
    time_metric = st.selectbox(
        "Select Metric for Time Series Analysis",
        ["total_rides", "incidents", "avg_braking_events", "avg_swerving_events"]
    )
    decomposition_fig = analyze_time_series(time_series_data, time_metric)
    st.plotly_chart(decomposition_fig, use_container_width=True)
    st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)
    anomaly_fig, anomalies = detect_anomalies(time_series_data.set_index('date'), column=time_metric)
    st.plotly_chart(anomaly_fig, use_container_width=True)
    st.subheader("Detected Anomalies")
    st.dataframe(anomalies.reset_index()[['date', time_metric, 'anomaly_score']])

with tab5:
    st.markdown('<div class="section-title">AI-Powered Safety Recommendations</div>', unsafe_allow_html=True)
    with st.spinner("Generating recommendations..."):
        recommendations_df = generate_safety_recommendations(braking_data, swerving_data)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card("High Priority Actions", len(recommendations_df[recommendations_df['Priority'] == 'High'])), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Infrastructure Needs", len(recommendations_df[recommendations_df['Type'] == 'Infrastructure'])), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Maintenance Needs", len(recommendations_df[recommendations_df['Type'] == 'Maintenance'])), unsafe_allow_html=True)
    st.dataframe(recommendations_df)
    st.markdown('<div class="section-title">Correlation Analysis</div>', unsafe_allow_html=True)
    numeric_cols = pd.concat([braking_data, swerving_data, route_data, time_series_data]).select_dtypes(include=[np.number])
    corr_fig = create_correlation_heatmap(numeric_cols)
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

st.markdown("""
<div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eaeaea;">
    <p style="color: #666; font-size: 14px;">© 2025 SeeSense Safety Analytics Platform. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
