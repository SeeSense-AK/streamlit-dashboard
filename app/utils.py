import pandas as pd
import numpy as np

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to lowercase and underscores.
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def filter_by_event_type(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """
    Filters the DataFrame by the specified event type.
    """
    return df[df['event_type'] == event_type]

def safe_number(val, default=0):
    """
    Converts value to float if possible, else returns default.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def format_latlon(lat, lon, precision=5):
    """
    Returns a tuple of (lat, lon) rounded to the given precision.
    """
    return (round(float(lat), precision), round(float(lon), precision))

def get_top_n(df: pd.DataFrame, col: str, n=5):
    """
    Returns top n rows of DataFrame sorted by column col (descending).
    """
    return df.sort_values(col, ascending=False).head(n)

def percent_change(new, old):
    """
    Computes percent change from old to new. Returns np.nan if old is zero.
    """
    try:
        if old == 0:
            return np.nan
        return (new - old) / old * 100
    except Exception:
        return np.nan

def drop_na_rows(df: pd.DataFrame, cols=None):
    """
    Drops rows with NA in specified columns, or all if cols is None.
    """
    return df.dropna(subset=cols) if cols else df.dropna()

def get_unique_values(df: pd.DataFrame, col: str):
    """
    Returns sorted list of unique values in a column.
    """
    return sorted(df[col].dropna().unique())

def to_datetime(df: pd.DataFrame, col: str):
    """
    Converts a DataFrame column to datetime if not already.
    """
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def safe_divide(a, b):
    """
    Divides a by b, returns np.nan if b is zero.
    """
    try:
        return a / b if b != 0 else np.nan
    except Exception:
        return np.nan
