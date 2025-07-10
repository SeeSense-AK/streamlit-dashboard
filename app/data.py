import pandas as pd
import os
from utils import clean_column_names, to_datetime

# Define the data directory relative to this script
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data() -> pd.DataFrame:
    """
    Loads and concatenates cycling data from predefined CSV files in the data/ directory.
    Standardizes column names and parses date columns.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame containing all cycling data.
    
    Raises:
        FileNotFoundError: If any of the required CSV files are missing.
        ValueError: If the loaded data is empty or missing required columns.
    """
    # Define paths to the CSV files
    required_files = {
        'route_data': os.path.join(DATA_DIR, 'route_data.csv'),
        'braking_data': os.path.join(DATA_DIR, 'braking_data.csv'),
        'swerving_data': os.path.join(DATA_DIR, 'swerving_data.csv'),
        'time_series_data': os.path.join(DATA_DIR, 'time_series_data.csv')
    }
    
    # Check if all files exist
    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
    
    # Load CSVs into DataFrames
    try:
        route_data = pd.read_csv(required_files['route_data'])
        braking_data = pd.read_csv(required_files['braking_data'])
        swerving_data = pd.read_csv(required_files['swerving_data'])
        time_series_data = pd.read_csv(required_files['time_series_data'], parse_dates=['date'])
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {str(e)}")
    
    # Clean column names for consistency
    route_data = clean_column_names(route_data)
    braking_data = clean_column_names(braking_data)
    swerving_data = clean_column_names(swerving_data)
    time_series_data = clean_column_names(time_series_data)
    
    # Attempt to parse any additional date/time columns (excluding 'date' which is already parsed)
    for df in [route_data, braking_data, swerving_data, time_series_data]:
        for col in df.columns:
            if col != 'date' and ('date' in col.lower() or 'time' in col.lower()):
                df = to_datetime(df, col)
    
    # Concatenate all DataFrames
    try:
        data = pd.concat([route_data, braking_data, swerving_data, time_series_data], ignore_index=True)
    except Exception as e:
        raise ValueError(f"Error concatenating DataFrames: {str(e)}")
    
    # Check if the concatenated DataFrame is empty
    if data.empty:
        raise ValueError("Concatenated DataFrame is empty. Check the input CSV files.")
    
    # Validate required columns
    required_columns = [
        'event_type', 'lat', 'lon', 'intensity', 'route_id', 
        'popularity_rating', 'start_lat', 'start_lon', 'end_lat', 'end_lon'
    ]
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {', '.join(missing_cols)}")
    
    return data

def filter_date_range(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    """
    Filters the DataFrame for rows within the start_date and end_date (inclusive).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of the date column to filter on.
        start_date: Start date for filtering (inclusive).
        end_date: End date for filtering (inclusive).
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    
    Raises:
        ValueError: If date_col is not in DataFrame or is not datetime.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"Column '{date_col}' is not in datetime format.")
    
    mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
    return df.loc[mask]

def get_event_types(df: pd.DataFrame) -> list:
    """
    Returns a sorted list of unique event types in the data.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        list: Sorted list of unique event types.
    """
    if 'event_type' in df.columns:
        return sorted(df['event_type'].dropna().unique())
    return []

def summarize_by_event(df: pd.DataFrame, group_cols: list = None) -> pd.DataFrame:
    """
    Summarizes event counts by specified columns (default: event_type).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        group_cols (list, optional): Columns to group by. Defaults to ['event_type'].
    
    Returns:
        pd.DataFrame: Summary DataFrame with event counts.
    """
    if group_cols is None:
        group_cols = ['event_type']
    if not all(col in df.columns for col in group_cols):
        raise ValueError(f"One or more group columns not found in DataFrame: {group_cols}")
    return df.groupby(group_cols).size().reset_index(name='count')
