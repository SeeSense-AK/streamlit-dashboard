import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import geopandas as gpd
from shapely.geometry import Point, LineString
import altair as alt
import os
from PIL import Image
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Cycling Safety Analytics Platform",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def load_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e89ae;
        color: white;
    }
    .css-1v3fvcr {
        background-color: #f9f9f9;
    }
    .st-bx {
        background-color: #ffffff;
    }
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 10rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
        padding-right: 1rem;
        padding-bottom: 3.5rem;
        padding-left: 1rem;
    }
    .st-bc {
        background-color: #f0f2f6;
    }
    .css-12oz5g7 {
        padding-top: 2rem;
        padding-right: 1rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .card-title {
        color: #555;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .card-value {
        color: #333;
        font-size: 24px;
        font-weight: 700;
    }
    .card-delta {
        color: #28a745;
        font-size: 14px;
        font-weight: 500;
    }
    .card-delta-down {
        color: #dc3545;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #333;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid #eaeaea;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .widget-label {
        font-weight: 600;
        margin-bottom: 5px;
        color: #555;
    }
    .btn-custom {
        background-color: #4e89ae;
        color: white;
        padding: 10px 15px;
        border-radius: 4px;
        cursor: pointer;
        text-align: center;
        width: 100%;
        margin-top: 10px;
        font-weight: 500;
    }
    .btn-custom:hover {
        background-color: #3a6d8a;
    }
    </style>
    """, unsafe_allow_html=True)

# Load sample data (replace with your actual data loading logic)
@st.cache_data
def load_data():
    # Sample route popularity data
    route_data = pd.DataFrame({
        'route_id': range(1, 5001),
        'start_lat': np.random.uniform(51.5, 51.6, 5000),
        'start_lon': np.random.uniform(-0.15, -0.05, 5000),
        'end_lat': np.random.uniform(51.5, 51.6, 5000),
        'end_lon': np.random.uniform(-0.15, -0.05, 5000),
        'distinct_cyclists': np.random.randint(10, 1000, 5000),
        'days_active': np.random.randint(1, 365, 5000),
        'popularity_rating': np.random.randint(1, 10, 5000),
        'avg_speed': np.random.uniform(10, 25, 5000),
        'avg_duration': np.random.uniform(5, 60, 5000),
        'route_type': np.random.choice(['Commute', 'Leisure', 'Exercise', 'Mixed'], 5000),
        'has_bike_lane': np.random.choice([True, False], 5000),
    })

    # Sample hotspot data for braking with controlled values and explicit types
    braking_data = pd.DataFrame({
        'hotspot_id': [f"BRK{i:03d}" for i in range(1, 251)],  # String IDs
        'lat': np.random.uniform(51.5, 51.6, 250),  # Keep as float
        'lon': np.random.uniform(-0.15, -0.05, 250),  # Keep as float
        'intensity': np.random.uniform(1, 10, 250),  # Keep as float
        'incidents_count': np.random.randint(5, 100, 250),
        'avg_deceleration': np.random.uniform(2, 8, 250),
        'road_type': np.random.choice(['Junction', 'Crossing', 'Roundabout', 'Straight'], 250),
        'surface_quality': [str(x) for x in range(1, 251)],  # Convert to strings
        'date_recorded': pd.date_range(end=pd.Timestamp.now(), periods=250, freq='D')
    })

    # Sample hotspot data for swerving with controlled values and explicit types
    swerving_data = pd.DataFrame({
        'hotspot_id': [f"SWV{i:03d}" for i in range(1, 251)],  # String IDs
        'lat': np.random.uniform(51.5, 51.6, 250),  # Keep as float
        'lon': np.random.uniform(-0.15, -0.05, 250),  # Keep as float
        'intensity': np.random.uniform(1, 10, 250),  # Keep as float
        'incidents_count': np.random.randint(5, 100, 250),
        'avg_lateral_movement': np.random.uniform(0.5, 3, 250),
        'road_type': np.random.choice(['Junction', 'Crossing', 'Roundabout', 'Straight'], 250),
        'obstruction_present': np.random.choice(['Yes', 'No'], 250),
        'date_recorded': pd.date_range(end=pd.Timestamp.now(), periods=250, freq='D')
    })

    # Time series data with explicit types
    date_range = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=365), 
                              end=pd.Timestamp.now(), 
                              freq='D')
    time_series_data = pd.DataFrame({
        'date': date_range,
        'total_rides': np.random.normal(1000, 200, len(date_range)).astype(int),
        'incidents': np.random.normal(50, 15, len(date_range)).astype(int),
        'avg_speed': np.random.normal(18, 3, len(date_range)),
        'avg_braking_events': np.random.normal(30, 8, len(date_range)),
        'avg_swerving_events': np.random.normal(25, 7, len(date_range)),
        'precipitation_mm': np.random.exponential(2, len(date_range)),
        'temperature': np.random.normal(15, 8, len(date_range))
    })

    return route_data, braking_data, swerving_data, time_series_data

# Logo and branding
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def add_logo():
    image_path = "/Users/abhishekkumbhar/Documents/RAMMSI_Data/streamlit_project/images/logo.png"  
    encoded_image = load_image(image_path)
    
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <img src="data:image/png;base64,{encoded_image}" width="150px">
    </div>
    <hr>
    """, unsafe_allow_html=True)

# Helper function for metric cards
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

# Create a plotly map for hotspots
def create_hotspot_map(data, color_scale, title, zoom=12):
    fig = px.scatter_map(
        data,
        lat="lat",
        lon="lon",
        size="intensity",
        color="intensity",
        color_continuous_scale=color_scale,
        size_max=20,
        zoom=zoom,
        map_style="carto-positron",
        hover_name="hotspot_id",
        hover_data={
            "lat": False,
            "lon": False,
            "intensity": True,
            "incidents_count": True,
            "road_type": True
        },
        title=title
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(
            title="Intensity",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300
        ),
        height=600
    )
    
    return fig

# Create a route popularity map
def create_route_map(data, zoom=12):
    # Create linestrings for routes
    routes = []
    for _, row in data.iterrows():
        routes.append({
            'route_id': int(row['route_id']),
            'popularity': float(row['popularity_rating']),
            'cyclists': int(row['distinct_cyclists']),
            'days': int(row['days_active']),
            'path': [[float(row['start_lon']), float(row['start_lat'])], 
                    [float(row['end_lon']), float(row['end_lat'])]],
            'color': [
                min(255, int(row['popularity_rating'] * 25)),  # More red for popular routes
                max(100, int(255 - row['popularity_rating'] * 20)),  # Less green for popular routes
                max(50, int(255 - row['popularity_rating'] * 25))   # Less blue for popular routes
            ]
        })
    
    view_state = pdk.ViewState(
        latitude=float(data['start_lat'].mean()),
        longitude=float(data['start_lon'].mean()),
        zoom=zoom,
        pitch=45
    )
    
    layer = pdk.Layer(
        "PathLayer",
        data=routes,
        get_path="path",
        get_width="popularity * 0.5",  # Reduced width multiplier
        get_color="color",
        width_scale=10,  # Reduced width scale
        width_min_pixels=1,  # Minimum width
        width_max_pixels=5,  # Maximum width
        rounded=True,  # Rounded line ends
        pickable=True,
        auto_highlight=True
    )
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Route ID:</b> {route_id}<br>"
                   "<b>Popularity:</b> {popularity}/10<br>"
                   "<b>Distinct Cyclists:</b> {cyclists}<br>"
                   "<b>Active Days:</b> {days}",
            "style": {
                "backgroundColor": "white",
                "color": "black"
            }
        }
    )

# ML models and predictions
@st.cache_resource
def train_risk_prediction_model(data):
    # Simple model to predict risk based on route features
    X = data[['distinct_cyclists', 'popularity_rating', 'avg_speed', 'has_bike_lane']].copy()
    X['has_bike_lane'] = X['has_bike_lane'].astype(int)
    
    # Create synthetic risk scores for demonstration
    y = (10 - data['popularity_rating']) * 0.3 + \
        (data['avg_speed'] > 20).astype(int) * 0.4 + \
        (1 - data['has_bike_lane'].astype(int)) * 0.3
    y = y * 10  # Scale to 0-10
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y.round().astype(int))
    
    return model

# Time series analysis
def analyze_time_series(data, column='incidents'):
    # Ensure data is sorted by date
    data = data.sort_values('date')
    data = data.set_index('date')
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data[column], model='additive', period=7)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=4, 
        cols=1,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Add traces
    fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
    
    fig.update_layout(height=800, title_text=f"Time Series Decomposition of {column.title()}")
    
    return fig

# Anomaly detection for safety incidents
def detect_anomalies(data, column='incidents', contamination=0.05):
    # Reshape data for isolation forest
    X = data[column].values.reshape(-1, 1)
    
    # Train isolation forest
    model = IsolationForest(contamination=contamination, random_state=42)
    data['anomaly'] = model.fit_predict(X)
    data['anomaly_score'] = model.decision_function(X)
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add normal points
    fig.add_trace(go.Scatter(
        x=data.index[data['anomaly'] == 1],
        y=data[column][data['anomaly'] == 1],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=6)
    ))
    
    # Add anomaly points
    fig.add_trace(go.Scatter(
        x=data.index[data['anomaly'] == -1],
        y=data[column][data['anomaly'] == -1],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title=f"Anomaly Detection for {column.title()}",
        xaxis_title="Date",
        yaxis_title=column.title(),
        legend_title="Data Points",
        height=500
    )
    
    return fig, data[data['anomaly'] == -1]

# Correlation analysis
def create_correlation_heatmap(data):
    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(height=600)
    
    return fig

# Safety recommendations generator
def generate_safety_recommendations(braking_data, swerving_data):
    # Identify top Priority areas
    braking_hotspots = braking_data.sort_values('intensity', ascending=False).head(5)
    swerving_hotspots = swerving_data.sort_values('intensity', ascending=False).head(5)
    
    recommendations = []
    
    # Generate recommendations for braking hotspots
    for _, hotspot in braking_hotspots.iterrows():
        if hotspot['road_type'] == 'Junction':
            recommendations.append({
                'Location': f"Junction at coordinates ({hotspot['lat']:.4f}, {hotspot['lon']:.4f})",
                'Issue': f"High intensity braking ({hotspot['intensity']:.1f}/10)",
                'Recommendation': "Consider installing advanced warning signs and reviewing junction visibility",
                'Priority': 'High' if hotspot['intensity'] > 7 else 'Medium',
                'Type': 'Infrastructure',
                'ROI Estimate': 'High'
            })
        elif hotspot['road_type'] == 'Crossing':
            recommendations.append({
                'Location': f"Crossing at coordinates ({hotspot['lat']:.4f}, {hotspot['lon']:.4f})",
                'Issue': f"Frequent sudden braking ({hotspot['incidents_count']} incidents)",
                'Recommendation': "Review crossing design and consider raised table crossing",
                'Priority': 'High' if hotspot['incidents_count'] > 50 else 'Medium',
                'Type': 'Infrastructure',
                'ROI Estimate': 'Medium'
            })
        else:
            recommendations.append({
                'Location': f"{hotspot['road_type']} at coordinates ({hotspot['lat']:.4f}, {hotspot['lon']:.4f})",
                'Issue': f"Sudden braking required (avg. deceleration {hotspot['avg_deceleration']:.1f} m/s²)",
                'Recommendation': "Investigate road surface quality and sight lines",
                'Priority': 'Medium',
                'Type': 'Maintenance',
                'ROI Estimate': 'Medium'
            })
    
    # Generate recommendations for swerving hotspots
    for _, hotspot in swerving_hotspots.iterrows():
        if hotspot['obstruction_present']:
            recommendations.append({
                'Location': f"{hotspot['road_type']} at coordinates ({hotspot['lat']:.4f}, {hotspot['lon']:.4f})",
                'Issue': f"Swerving due to obstructions (lateral movement {hotspot['avg_lateral_movement']:.1f}m)",
                'Recommendation': "Remove or redesign road furniture, widen cycle lane",
                'Priority': 'High' if hotspot['intensity'] > 7 else 'Medium',
                'Type': 'Infrastructure',
                'ROI Estimate': 'High'
            })
        else:
            recommendations.append({
                'Location': f"{hotspot['road_type']} at coordinates ({hotspot['lat']:.4f}, {hotspot['lon']:.4f})",
                'Issue': f"Unexpected swerving ({hotspot['incidents_count']} incidents)",
                'Recommendation': "Investigate surface quality and drainage issues",
                'Priority': 'Medium',
                'Type': 'Maintenance',
                'ROI Estimate': 'Medium'
            })
    
    return pd.DataFrame(recommendations)

# Main application
def main():
    # Load CSS
    load_css()
    
    # Add logo
    add_logo()
    
    # Load data
    route_data, braking_data, swerving_data, time_series_data = load_data()
    
    # Sidebar - Filters and controls
    st.sidebar.markdown('<div class="widget-label">Data Filters</div>', unsafe_allow_html=True)
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        # Ensure start_date and end_date are datetime objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        # Filter the time_series_data based on the selected date range
        time_series_data = time_series_data[(time_series_data['date'] >= start_date) & (time_series_data['date'] <= end_date)]
    
    min_popularity = st.sidebar.slider(
        "Min. Route Popularity",
        min_value=1,
        max_value=10,
        value=1
    )
    
    min_cyclists = st.sidebar.slider(
        "Min. Distinct Cyclists",
        min_value=10,
        max_value=1000,
        value=50
    )
    
    # Apply filters
    filtered_routes = route_data[
        (route_data['popularity_rating'] >= min_popularity) &
        (route_data['distinct_cyclists'] >= min_cyclists)
    ]
    
    # Dashboard tiers
    st.sidebar.markdown('<div class="widget-label">Dashboard Tier</div>', unsafe_allow_html=True)
    dashboard_tier = st.sidebar.selectbox(
        "Select Your Subscription",
        options=["Standard", "Premium", "Ultimate"],
        index=2  # Default to Ultimate
    )
    
    # Export options
    st.sidebar.markdown('<div class="widget-label">Export Options</div>', unsafe_allow_html=True)
    export_format = st.sidebar.selectbox(
        "Export Format",
        options=["PDF Report", "Excel", "CSV", "GeoJSON"]
    )
    
    st.sidebar.markdown('<div class="btn-custom">Export Dashboard</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)

    # Help and support
    with st.sidebar.expander("Help & Support"):
        st.write("Need assistance with your dashboard? Contact our support team:")
        st.write("📧 support@seesense.cc")
        st.write("📞 028 9107 8353")
    
    # Main content area
    st.title("SeeSense Safety Analytics Platform")
    
    # Tabs for different sections
    tabs = st.tabs([
        "📊 Dashboard Overview", 
        "🔍 ML Insights", 
        "🗺️ Spatial Analysis", 
        "📈 Advanced Analytics",
        "💡 Actionable Insights"
    ])
    
    # Tab 1: Dashboard Overview
    with tabs[0]:
        st.subheader("Key Safety Metrics")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                metric_card(
                    "Total Routes Analyzed", 
                    f"{len(filtered_routes):,}", 
                    "+5% vs prev month", 
                    "positive"
                ), 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                metric_card(
                    "Braking Hotspots", 
                    f"{len(braking_data)}", 
                    "-12% vs prev month", 
                    "positive"
                ), 
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                metric_card(
                    "Swerving Hotspots", 
                    f"{len(swerving_data)}", 
                    "-8% vs prev month", 
                    "positive"
                ), 
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                metric_card(
                    "Safety Score", 
                    "8.4", 
                    "+0.6 vs prev month", 
                    "positive",
                    "/10"
                ), 
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Maps section
        st.markdown('<div class="section-title">Safety Hotspot Maps</div>', unsafe_allow_html=True)
        
        map_col1, map_col2 = st.columns(2)
        
        with map_col1:
            st.plotly_chart(
                create_hotspot_map(
                    braking_data, 
                    "YlOrRd", 
                    "Braking Hotspots"
                ),
                use_container_width=True
            )
        
        with map_col2:
            st.plotly_chart(
                create_hotspot_map(
                    swerving_data, 
                    "PuRd", 
                    "Swerving Hotspots"
                ),
                use_container_width=True
            )
        
        # Route popularity map
        st.markdown('<div class="section-title">Route Popularity Map</div>', unsafe_allow_html=True)

        # Add map controls
        map_zoom = st.slider("Map Zoom Level", 10, 15, 12)
        sample_size = st.slider("Number of Routes to Display", 100, 2000, 1000)

        # Create map with filtered and sampled data
        filtered_routes_sample = filtered_routes.sample(min(sample_size, len(filtered_routes))) if not filtered_routes.empty else pd.DataFrame()

        if filtered_routes_sample.empty:
            st.warning("No routes match the selected criteria. Try adjusting the filters.")
        else:
            try:
                deck = create_route_map(filtered_routes_sample, zoom=map_zoom)
                # Display the map
                st.pydeck_chart(deck)
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
                st.info("Try adjusting the filters or reducing the minimum popularity score.")
        
        # Time series overview
        st.markdown('<div class="section-title">Incident Trends (Last 30 Days)</div>', unsafe_allow_html=True)
        
        # Filter time series to last 30 days
        recent_time_series = time_series_data.tail(30).copy()
        
        # Create line chart with multiple metrics
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recent_time_series['date'],
            y=recent_time_series['incidents'],
            mode='lines',
            name='Safety Incidents',
            line=dict(color='#e41a1c', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_time_series['date'],
            y=recent_time_series['avg_braking_events'],
            mode='lines',
            name='Avg. Braking Events',
            line=dict(color='#377eb8', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_time_series['date'],
            y=recent_time_series['avg_swerving_events'],
            mode='lines',
            name='Avg. Swerving Events',
            line=dict(color='#4daf4a', width=2)
        ))
        
        fig.update_layout(
            title="Safety Incidents and Events Trend",
            xaxis_title="Date",
            yaxis_title="Count",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: ML Insights
    with tabs[1]:
        st.subheader("Machine Learning Insights")
        
        ml_col1, ml_col2 = st.columns([2, 1])
        
        with ml_col1:
            st.markdown('<div class="section-title">Risk Prediction Model</div>', unsafe_allow_html=True)
            
            # Train risk prediction model
            risk_model = train_risk_prediction_model(route_data)
            
            # Get feature importances
            feature_importance = pd.DataFrame({
                'Feature': ['Traffic Signal', 'Post box', 'Schools', 'Ped Crossing'],
                'Importance': risk_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                color='Importance',
                color_continuous_scale='Blues',
                title="Risk Factors Importance"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Example predictions
            st.markdown('<div class="section-title">Route Safety Predictions</div>', unsafe_allow_html=True)
            
            # Make predictions on a sample of routes
            X_sample = route_data.sample(10)[['distinct_cyclists', 'popularity_rating', 'avg_speed', 'has_bike_lane']].copy()
            X_sample['has_bike_lane'] = X_sample['has_bike_lane'].astype(int)
            
            # Predict risk level
            risk_predictions = risk_model.predict(X_sample)
            
            # Create a dataframe with the sample and predictions
            prediction_df = pd.DataFrame({
                'Route ID': route_data.loc[X_sample.index, 'route_id'].values,
                'Distinct Cyclists': X_sample['distinct_cyclists'].values,
                'Popularity': X_sample['popularity_rating'].values,
                'Avg. Speed (km/h)': X_sample['avg_speed'].values,
                'Has Bike Lane': X_sample['has_bike_lane'].values,
                'Predicted Risk Level': risk_predictions
            })
            
            # Style the dataframe
            def color_risk(val):
                if val >= 7:
                    return 'background-color: #ffcccc'
                elif val >= 4:
                    return 'background-color: #ffffcc'
                else:
                    return 'background-color: #ccffcc'
            
            styled_predictions = prediction_df.style.applymap(
                color_risk, 
                subset=['Predicted Risk Level']
            )
            
            st.dataframe(styled_predictions, use_container_width=True)
        
        with ml_col2:
            st.markdown('<div class="section-title">Behavioral Clusters</div>', unsafe_allow_html=True)
            
            # Perform k-means clustering
            kmeans_data = route_data[['distinct_cyclists', 'popularity_rating', 'avg_speed', 'avg_duration']].copy()
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(kmeans_data)
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
            kmeans.fit(scaled_data)
            
            # Add cluster labels to the data
            route_data_clustered = route_data.copy()
            route_data_clustered['cluster'] = kmeans.labels_
            
            # Create cluster names
            cluster_names = {
                0: 'Commuters',
                1: 'Exercise Riders',
                2: 'Leisure Cyclists',
                3: 'Mixed Usage'
            }
            
            route_data_clustered['cluster_name'] = route_data_clustered['cluster'].map(cluster_names)
            
            # Create a scatter plot of clusters
            fig = px.scatter(
                route_data_clustered.sample(min(500, len(route_data_clustered))),
                x='avg_speed',
                y='avg_duration',
                color='cluster_name',
                size='distinct_cyclists',
                hover_data=['route_id', 'popularity_rating'],
                title="Cyclist Behavior Clusters",
                color_discrete_sequence=px.colors.qualitative.G10,
                labels={
                    'avg_speed': 'Average Speed (km/h)',
                    'avg_duration': 'Average Trip Duration (min)',
                    'cluster_name': 'Rider Type'
                }
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            cluster_stats = route_data_clustered.groupby('cluster_name').agg({
                'distinct_cyclists': 'mean',
                'popularity_rating': 'mean',
                'avg_speed': 'mean',
                'avg_duration': 'mean'
            }).round(1)
            
            st.markdown('<div class="section-title">Rider Segment Profiles</div>', unsafe_allow_html=True)
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Anomaly detection
            st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)
            
            # Description of the anomaly detection capability
            st.write("""
            Our anomaly detection system identifies unusual patterns in cyclist behavior that may 
            indicate emerging safety concerns or infrastructure issues.
            """)
            
            # Show a sample anomaly
            st.info("""
            **Detected Anomaly: Junction at (51.53, -0.12)**
            
            Unusual increase in braking incidents (+325%) detected on Thursday afternoons.
            Possible cause: Traffic signal timing issues during rush hour.
            """)
    
    # Tab 3: Spatial Analysis
    with tabs[2]:
        st.subheader("Spatial Analysis")
        
        spatial_col1, spatial_col2 = st.columns([3, 2])
        
        with spatial_col1:
            st.markdown('<div class="section-title">Density Heatmap Analysis</div>', unsafe_allow_html=True)

            heatmap_type = st.selectbox(
                "Select Heatmap Type",
                 options=["Braking Incidents", "Swerving Incidents", "Combined Safety Score"]
            )

            # Replace the folium map section with this:
            try:
                if (heatmap_type == "Braking Incidents"):
                    # Create heatmap using plotly
                    fig = px.density_mapbox(
                    braking_data, 
                    lat='lat', 
                    lon='lon', 
                    z='intensity',
                    radius=20,
                    center=dict(lat=51.55, lon=-0.1),
                    zoom=11,
                    mapbox_style="carto-positron",
                    title="Braking Incidents Density",
                    color_continuous_scale="Viridis"
                    )
    
                elif (heatmap_type == "Swerving Incidents"):
                    fig = px.density_mapbox(
                    swerving_data,
                    lat='lat',
                    lon='lon',
                    z='intensity',
                    radius=20,
                    center=dict(lat=51.55, lon=-0.1),
                    zoom=11,
                    mapbox_style="carto-positron",
                    title="Swerving Incidents Density",
                    color_continuous_scale="Viridis"
                    )
    
                else:
                    # Combine the data for combined view
                    combined_data = pd.concat([
                        braking_data[['lat', 'lon', 'intensity']],
                        swerving_data[['lat', 'lon', 'intensity']]
                    ])
        
                    fig = px.density_mapbox(
                        combined_data,
                        lat='lat',
                        lon='lon',
                        z='intensity',
                        radius=20,
                        center=dict(lat=51.55, lon=-0.1),
                        zoom=11,
                        mapbox_style="carto-positron",
                        title="Combined Safety Score Density",
                        color_continuous_scale="Viridis"
                    )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
                st.write("Please try refreshing the page or contact support if the Issue persists.")
            
            # Spatial correlation analysis
            st.markdown('<div class="section-title">Infrastructure Impact Analysis</div>', unsafe_allow_html=True)
            
            # Create fake infrastructure data for demonstration
            infrastructure_data = pd.DataFrame({
                'Type': ['Bike Lane', 'Shared Path', 'Painted Lane', 'Protected Lane', 'No Infrastructure'],
                'incident_rate': [2.3, 3.1, 4.7, 1.8, 8.2],
                'avg_severity': [3.2, 4.5, 5.1, 2.7, 6.8],
                'length_km': [45, 28, 35, 12, 80]
            })
            
            # Create a bubble chart
            fig = px.scatter(
                infrastructure_data,
                x='incident_rate',
                y='avg_severity',
                size='length_km',
                color='Type',
                title="Safety by Infrastructure Type",
                labels={
                    'incident_rate': 'Incidents per km per month',
                    'avg_severity': 'Average Incident Severity (1-10)',
                    'length_km': 'Total Length (km)',
                    'Type': 'Infrastructure Type'
                },
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with spatial_col2:
            # Route optimization
            st.markdown('<div class="section-title">Route Safety Optimization</div>', unsafe_allow_html=True)
            
            st.write("""
            Our route optimization algorithm analyzes safety data to recommend the safest cycling routes 
            between any two points, balancing safety with journey time.
            """)
            
            # Origins and destinations
            st.text_input("Starting Point", "51.5074, -0.1278 (Example)")
            st.text_input("Destination", "51.5311, -0.1221 (Example)")
            
            st.selectbox(
                "Optimization Priority",
                options=["Safety First", "Balanced", "Speed First"]
            )
            
            st.markdown('<div class="btn-custom">Find Safest Route</div>', unsafe_allow_html=True)
            
            # Static example of route comparison
            st.markdown('<div class="section-title">Route Comparison</div>', unsafe_allow_html=True)
            
            route_comparison = pd.DataFrame({
                'Route': ['Direct Route', 'Safety Optimized', 'Time Optimized'],
                'Distance (km)': [4.2, 4.8, 4.1],
                'Est. Time (min)': [18, 22, 17],
                'Safety Score': [5.8, 8.9, 6.2],
                'Risk Factors': ['High traffic exposure', 'Minimal traffic exposure', 'Moderate traffic exposure']
            })
            
            st.dataframe(route_comparison, use_container_width=True)
            
            # Terrain analysis
            st.markdown('<div class="section-title">Terrain Impact Analysis</div>', unsafe_allow_html=True)
            
            # Create elevation profile
            x = np.arange(0, 5, 0.1)
            elevation = 50 + 30 * np.sin(x) + np.random.normal(0, 2, len(x))
            
            # Simulate braking events correlated with downhill segments
            braking_events = np.zeros_like(x)
            braking_events[np.where(np.diff(elevation, prepend=elevation[0]) < -5)[0]] = np.random.randint(3, 15, size=len(np.where(np.diff(elevation, prepend=elevation[0]) < -5)[0]))
            
            # Create figure with dual y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add elevation trace
            fig.add_trace(
                go.Scatter(x=x, y=elevation, name="Elevation", line=dict(color='brown', width=2)),
                secondary_y=False,
            )
            
            # Add braking events trace
            fig.add_trace(
                go.Bar(x=x, y=braking_events, name="Braking Events", marker_color='red'),
                secondary_y=True,
            )
            
            # Set axis titles
            fig.update_xaxes(title_text="Distance (km)")
            fig.update_yaxes(title_text="Elevation (m)", secondary_y=False)
            fig.update_yaxes(title_text="Braking Events", secondary_y=True)
            
            fig.update_layout(
                title_text="Terrain Impact on Braking Behavior",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Advanced Analytics
    with tabs[3]:
        st.subheader("Advanced Analytics")
        
        # Time series analysis
        st.markdown('<div class="section-title">Temporal Pattern Analysis</div>', unsafe_allow_html=True)
        
        time_metric = st.selectbox(
            "Select Metric for Time Series Analysis",
            options=["Total Rides", "Safety Incidents", "Avg. Braking Events", "Avg. Swerving Events"],
            index=1
        )
        
        metric_mapping = {
            "Total Rides": "total_rides",
            "Safety Incidents": "incidents",
            "Avg. Braking Events": "avg_braking_events",
            "Avg. Swerving Events": "avg_swerving_events"
        }
        
        selected_metric = metric_mapping[time_metric]
        
        # Run time series decomposition
        decomposition_fig = analyze_time_series(time_series_data, selected_metric)
        st.plotly_chart(decomposition_fig, use_container_width=True)
        
        # Day of week and time of day analysis
        st.markdown('<div class="section-title">Day of Week & Time of Day Patterns</div>', unsafe_allow_html=True)
        
        analytics_col1, analytics_col2 = st.columns(2)
        
        with analytics_col1:
            # Create day of week aggregation
            dow_data = time_series_data.copy()
            dow_data['day_of_week'] = dow_data['date'].dt.day_name()
            dow_summary = dow_data.groupby('day_of_week').agg({
                'incidents': 'mean',
                'avg_braking_events': 'mean',
                'avg_swerving_events': 'mean'
            }).reset_index()
            
            # Ensure days are in correct order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_summary['day_of_week'] = pd.Categorical(dow_summary['day_of_week'], categories=day_order, ordered=True)
            dow_summary = dow_summary.sort_values('day_of_week')
            
            # Create bar chart
            fig = px.bar(
                dow_summary,
                x='day_of_week',
                y='incidents',
                title="Average Daily Incidents by Day of Week",
                color='incidents',
                color_continuous_scale=px.colors.sequential.Reds
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with analytics_col2:
            # Create simulated time of day data
            hours = list(range(24))
            incidents_by_hour = [
                5, 3, 2, 2, 3, 8, 20, 35, 25, 15, 12, 18, 
                22, 20, 18, 22, 35, 42, 28, 15, 10, 8, 7, 6
            ]
            
            # Create time of day data
            tod_data = pd.DataFrame({
                'hour': hours,
                'incidents': incidents_by_hour
            })
            
            # Create line chart
            fig = px.line(
                tod_data,
                x='hour',
                y='incidents',
                title="Average Incidents by Hour of Day",
                markers=True
            )
            
            # Add peak markers
            peak_hours = tod_data.nlargest(2, 'incidents')
            
            fig.add_trace(
                go.Scatter(
                    x=peak_hours['hour'],
                    y=peak_hours['incidents'],
                    mode='markers',
                    marker=dict(color='red', size=12),
                    name='Peak Hours'
                )
            )
            
            fig.update_layout(height=400)
            fig.update_xaxes(dtick=2)
            st.plotly_chart(fig, use_container_width=True)
        
        # Weather impact analysis
        st.markdown('<div class="section-title">Weather Impact Analysis</div>', unsafe_allow_html=True)
        
        analytics_col3, analytics_col4 = st.columns(2)
        
        with analytics_col3:
            # Create a scatter plot of incidents vs precipitation
            fig = px.scatter(
                time_series_data,
                x='precipitation_mm',
                y='incidents',
                title="Impact of Precipitation on Safety Incidents",
                trendline="ols",
                labels={
                    'precipitation_mm': 'Daily Precipitation (mm)',
                    'incidents': 'Number of Safety Incidents'
                }
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with analytics_col4:
            # Create a scatter plot of incidents vs temperature
            fig = px.scatter(
                time_series_data,
                x='temperature',
                y='incidents',
                title="Impact of Temperature on Safety Incidents",
                trendline="ols",
                labels={
                    'temperature': 'Temperature (°C)',
                    'incidents': 'Number of Safety Incidents'
                }
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly detection
        st.markdown('<div class="section-title">Anomaly Detection & Alerts</div>', unsafe_allow_html=True)
        
        anomaly_fig, anomalies = detect_anomalies(time_series_data, 'incidents')
        st.plotly_chart(anomaly_fig, use_container_width=True)
        
        # Display anomalies in a table
        if not anomalies.empty:
            st.subheader("Detected Anomalies")
            
            anomaly_display = anomalies.reset_index()[['date', 'incidents', 'anomaly_score']].copy()
            anomaly_display['anomaly_score'] = anomaly_display['anomaly_score'].round(3)
            anomaly_display.columns = ['Date', 'Incident Count', 'Anomaly Score']
            
            st.dataframe(anomaly_display, use_container_width=True)
            
            # Anomaly insights
            st.info("""
            **Anomaly Insights:**
            
            The detected anomalies coincide with severe weather events and public holidays. 
            The highest anomaly on May 25 corresponded with a major cycling event in the city.
            We recommend adding event data as a variable in future analysis.
            """)
    
    # Tab 5: Actionable Insights
    with tabs[4]:
        st.subheader("Actionable Insights & Recommendations")
        
        # Generate recommendations
        recommendations = generate_safety_recommendations(braking_data, swerving_data)
        
        # Summary metrics
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown(
                metric_card(
                    "High Priority Actions", 
                    f"{len(recommendations[recommendations['Priority'] == 'High'])}", 
                    None, 
                    None
                ), 
                unsafe_allow_html=True
            )
        
        with insight_col2:
            st.markdown(
                metric_card(
                    "Infrastructure Needs", 
                    f"{len(recommendations[recommendations['Type'] == 'Infrastructure'])}", 
                    None, 
                    None
                ), 
                unsafe_allow_html=True
            )
        
        with insight_col3:
            st.markdown(
                metric_card(
                    "Maintenance Needs", 
                    f"{len(recommendations[recommendations['Type'] == 'Maintenance'])}", 
                    None, 
                    None
                ), 
                unsafe_allow_html=True
            )
        
        # Safety recommendations
        st.markdown('<div class="section-title">Priority Safety Recommendations</div>', unsafe_allow_html=True)
        
        # Filter to high Priority
        high_priority = recommendations[recommendations['Priority'] == 'High'].reset_index(drop=True)
        
        # Display recommendations in cards
        for i in range(0, len(high_priority), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                if i < len(high_priority):
                    rec = high_priority.iloc[i]
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:15px; border-radius:5px; margin-bottom:10px; height:220px;">
                        <h4 style="color:#d9534f;">{rec['Location']}</h4>
                        <p><strong>Issue:</strong> {rec['Issue']}</p>
                        <p><strong>Recommendation:</strong> {rec['Recommendation']}</p>
                        <p><strong>Type:</strong> {rec['Type']} | <strong>ROI:</strong> {rec['ROI Estimate']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if i + 1 < len(high_priority):
                    rec = high_priority.iloc[i + 1]
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:15px; border-radius:5px; margin-bottom:10px; height:220px;">
                        <h4 style="color:#d9534f;">{rec['Location']}</h4>
                        <p><strong>Issue:</strong> {rec['Issue']}</p>
                        <p><strong>Recommendation:</strong> {rec['Recommendation']}</p>
                        <p><strong>Type:</strong> {rec['Type']} | <strong>ROI:</strong> {rec['ROI Estimate']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show all recommendations in a table
        st.markdown('<div class="section-title">All Safety Recommendations</div>', unsafe_allow_html=True)
        
        # Filter options
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            priority_filter = st.selectbox(
                "Filter by Priority",
                options=["All", "High", "Medium", "Low"],
                index=0
            )
        
        with rec_col2:
            type_filter = st.selectbox(
                "Filter by Type",
                options=["All", "Infrastructure", "Maintenance", "Education"],
                index=0
            )
        
        with rec_col3:
            roi_filter = st.selectbox(
                "Filter by ROI",
                options=["All", "High", "Medium", "Low"],
                index=0
            )
        
        # Apply filters
        filtered_recommendations = recommendations.copy()
        
        if priority_filter != "All":
            filtered_recommendations = filtered_recommendations[filtered_recommendations['Priority'] == priority_filter]
        
        if type_filter != "All":
            filtered_recommendations = filtered_recommendations[filtered_recommendations['Type'] == type_filter]
        
        if roi_filter != "All":
            filtered_recommendations = filtered_recommendations[filtered_recommendations['ROI Estimate'] == roi_filter]
        
        # Display filtered recommendations
        st.dataframe(filtered_recommendations, use_container_width=True)
        
        # ROI analysis
        st.markdown('<div class="section-title">Investment ROI Analysis</div>', unsafe_allow_html=True)
        
        roi_col1, roi_col2 = st.columns(2)
        
        with roi_col1:
            # Create simulated ROI data
            roi_data = pd.DataFrame({
                'investment_type': ['Junction Improvements', 'Protected Bike Lanes', 'Surface Repairs', 
                                  'Signage Updates', 'Lighting Improvements'],
                'cost_estimate': [120000, 450000, 85000, 35000, 70000],
                'incident_reduction': [32, 68, 25, 12, 18],
                'roi_percent': [22, 10, 24, 29, 21]
            })
            
            # Create ROI comparison chart
            fig = px.bar(
                roi_data,
                x='investment_type',
                y='roi_percent',
                color='cost_estimate',
                text='roi_percent',
                title="ROI by Investment Type",
                labels={
                    'investment_type': 'Investment Type',
                    'roi_percent': 'Estimated ROI (%)',
                    'cost_estimate': 'Estimated Cost ($)'
                },
                color_continuous_scale='Viridis'
            )
            
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with roi_col2:
            # Create impact analysis chart
            fig = px.scatter(
                roi_data,
                x='cost_estimate',
                y='incident_reduction',
                size='roi_percent',
                color='investment_type',
                title="Cost vs. Incident Reduction",
                labels={
                    'cost_estimate': 'Estimated Cost ($)',
                    'incident_reduction': 'Estimated Incident Reduction (%)',
                    'roi_percent': 'ROI (%)',
                    'investment_type': 'Investment Type'
                }
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Natural language insights summary
        st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Key Findings and Recommendations
        
        Based on our comprehensive analysis of cycling safety data, we've identified several key insights and high-impact recommendations:
        
        1. **Junction Safety**: Four of the top ten braking hotspots are at junctions with poor visibility. Installing advanced warning signs and improving sight lines could reduce incidents by up to 32%.
        
        2. **Weather Impact**: Rainy conditions correlate with a 28% increase in swerving incidents. Targeted infrastructure improvements in high-traffic areas could significantly improve safety during inclement weather.
        
        3. **Time-based Patterns**: Incident rates peak during morning (8-9 AM) and evening (5-6 PM) commute hours, particularly on Tuesdays and Thursdays. Targeted enforcement and awareness campaigns during these windows could yield substantial safety improvements.
        
        4. **Infrastructure ROI**: Protected bike lane investments show the highest absolute incident reduction (68%), while signage updates provide the highest percentage ROI (34%). A balanced approach combining both strategies is recommended.
        
        5. **Behavioral Insights**: "Commuter" cyclists experience different safety challenges than "Exercise" cyclists, with the former encountering more junction-related incidents and the latter more affected by road surface issues. Tailored safety messaging for each segment could enhance effectiveness.
        
        Our ML prediction model suggests that implementing the high-Priority recommendations could improve the overall safety score by 1.2 points (from 8.4 to 9.6) within 6 months of implementation.
        """)
        
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eaeaea;">
        <p style="color: #666; font-size: 14px;">© 2025 SeeSense Safety Analytics Platform. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

