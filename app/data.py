import pandas as pd
from app.utils import clean_column_names, to_datetime

def load_data(filepath: str = "data/sample_cycling_data.csv") -> pd.DataFrame:
    """
    Loads cycling data from a CSV file, cleans column names, and parses dates.
    Default is the sample data in the data/ directory.
    """
    df = pd.read_csv(filepath)
    df = clean_column_names(df)
    # Attempt to parse common date/time columns
    for col in df.columns:
        if "date" in col or "time" in col:
            df = to_datetime(df, col)
    return df

def load_multiple_csv(files: list) -> pd.DataFrame:
    """
    Loads and concatenates multiple CSV files into one DataFrame.
    """
    df_list = [load_data(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def filter_date_range(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    """
    Filters the DataFrame for rows within the start_date and end_date (inclusive).
    """
    mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
    return df.loc[mask]

def get_event_types(df: pd.DataFrame) -> list:
    """
    Returns a sorted list of unique event types in the data.
    """
    if "event_type" in df.columns:
        return sorted(df["event_type"].dropna().unique())
    return []

def summarize_by_event(df: pd.DataFrame, group_cols: list = None) -> pd.DataFrame:
    """
    Summarizes event counts by specified columns (default: event_type).
    """
    if group_cols is None:
        group_cols = ["event_type"]
    return df.groupby(group_cols).size().reset_index(name="count")
