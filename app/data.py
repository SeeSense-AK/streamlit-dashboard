import pandas as pd
import os
from utils import clean_column_names, to_datetime

# Define the data directory relative to this script
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data() -> dict:
    """
    Loads cycling data from predefined CSV files in the data/ directory.
    Standardizes column names and parses date columns.
    
    Returns:
        dict: Dictionary containing DataFrames for 'route_data', 'braking_data', 
              'swerving_data', and 'time_series_data'.
    
    Raises:
        FileNotFoundError: If any of the required CSV files are missing.
        ValueError: If any DataFrame is empty or missing required columns.
    """
    data_files = {
        'route_data': {
            'path': os.path.join(DATA_DIR, 'route_data.csv'),
            'required_columns': [
                'route_id', 'start_lat', 'start_lon', 'end_lat', 'end_lon',
                'distinct_cyclists', 'days_active', 'popularity_rating',
                'avg_speed', 'avg_duration', 'route_type', 'has_bike_lane'
            ]
        },
        'braking_data': {
            'path': os.path.join(DATA_DIR, 'braking_data.csv'),
            'required_columns': [
                'hotspot_id', 'lat', 'lon', 'intensity', 'incidents_count',
                'avg_deceleration', 'road_type', 'surface_quality', 'date_recorded'
            ]
        },
        'swerving_data': {
            'path': os.path.join(DATA_DIR, 'swerving_data.csv'),
            'required_columns': [
                'hotspot_id', 'lat', 'lon', 'intensity', 'incidents_count',
                'avg_lateral_movement', 'road_type', 'obstruction_present', 'date_recorded'
            ]
        },
        'time_series_data': {
            'path': os.path.join(DATA_DIR, 'time_series_data.csv'),
            'required_columns': [
                'date', 'total_rides', 'incidents', 'avg_speed',
                'avg_braking_events', 'avg_swerving_events', 'precipitation_mm', 'temperature'
            ]
        }
    }
    
    for name, info in data_files.items():
        if not os.path.exists(info['path']):
            raise FileNotFoundError(f"Data file not found: {info['path']}")
    
    data_dict = {}
    try:
        for name, info in data_files.items():
            if name == 'time_series_data':
                df = pd.read_csv(info['path'], parse_dates=['date'])
            else:
                df = pd.read_csv(info['path'])
            
            df = clean_column_names(df)
            
            missing_cols = [col for col in info['required_columns'] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {name}: {', '.join(missing_cols)}")
            
            if df.empty:
                raise ValueError(f"{name} is empty. Check the CSV file: {info['path']}")
            
            for col in df.columns:
                if col != 'date' and ('date' in col.lower() or 'time' in col.lower()):
                    df = to_datetime(df, col)
            
            data_dict[name] = df
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {str(e)}")
    
    return data_dict

def filter_date_range(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"Column '{date_col}' is not in datetime format.")
    
    mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
    return df.loc[mask]

def get_event_types(df: pd.DataFrame) -> list:
    if 'event_type' in df.columns:
        return sorted(df['event_type'].dropna().unique())
    return []

def summarize_by_event(df: pd.DataFrame, group_cols: list = None) -> pd.DataFrame:
    if group_cols is None:
        if 'event_type' in df.columns:
            group_cols = ['event_type']
        elif 'hotspot_id' in df.columns:
            group_cols = ['hotspot_id']
        elif 'route_id' in df.columns:
            group_cols = ['route_id']
        else:
            group_cols = [df.columns[0]]
    if not all(col in df.columns for col in group_cols):
        raise ValueError(f"One or more group columns not found in DataFrame: {group_cols}")
    return df.groupby(group_cols).size().reset_index(name='count')
