import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def create_hotspot_map(data, color_scale, title, zoom=12):
    fig = px.scatter_mapbox(
        data,
        lat="lat",
        lon="lon",
        size="intensity",
        color="intensity",
        color_continuous_scale=color_scale,
        size_max=20,
        zoom=zoom,
        mapbox_style="carto-positron",
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

def create_route_map(data, zoom=12):
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
                min(255, int(row['popularity_rating'] * 25)),
                max(100, int(255 - row['popularity_rating'] * 20)),
                max(50, int(255 - row['popularity_rating'] * 25))
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
        get_width="popularity * 0.5",
        get_color="color",
        width_scale=10,
        width_min_pixels=1,
        width_max_pixels=5,
        rounded=True,
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

def analyze_time_series(data, column='incidents'):
    data = data.sort_values('date')
    data = data.set_index('date')
    decomposition = seasonal_decompose(data[column], model='additive', period=7)
    fig = make_subplots(
        rows=4, 
        cols=1,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
    fig.update_layout(height=800, title_text=f"Time Series Decomposition of {column.title()}")
    return fig

def detect_anomalies(data, column='incidents', contamination=0.05):
    X = data[column].values.reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=42)
    data['anomaly'] = model.fit_predict(X)
    data['anomaly_score'] = model.decision_function(X)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index[data['anomaly'] == 1],
        y=data[column][data['anomaly'] == 1],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=6)
    ))
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

def create_correlation_heatmap(data):
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Matrix"
    )
    fig.update_layout(height=600)
    return fig
