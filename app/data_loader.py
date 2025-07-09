import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data():
    route_path = os.path.join(DATA_DIR, 'route_data.csv')
    braking_path = os.path.join(DATA_DIR, 'braking_data.csv')
    swerving_path = os.path.join(DATA_DIR, 'swerving_data.csv')
    timeseries_path = os.path.join(DATA_DIR, 'time_series_data.csv')
    
    # Load CSVs into DataFrames
    route_data = pd.read_csv(route_path)
    braking_data = pd.read_csv(braking_path)
    swerving_data = pd.read_csv(swarming_path)
    time_series_data = pd.read_csv(timeseries_path, parse_dates=['date'])  # parses 'date' as datetime

    return route_data, braking_data, swerving_data, time_series_data
