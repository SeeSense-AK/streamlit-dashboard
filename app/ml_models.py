import joblib
import pandas as pd
import os
import xgboost as xgb
import streamlit as st

def load_xgb_model():
    """
    Load the pretrained XGBoost model from the assets directory.
    
    Returns:
        xgb.Booster or xgb.XGBClassifier: Loaded XGBoost model.
    
    Raises:
        FileNotFoundError: If xgb_model.pkl is not found.
        Exception: For other loading errors.
    """
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'xgb_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading XGBoost model: {str(e)}")
        raise

def predict_risk(model, X: pd.DataFrame) -> list:
    """
    Predict risk levels using the loaded XGBoost model.
    
    Args:
        model: Loaded XGBoost model (xgb.Booster or xgb.XGBClassifier).
        X (pd.DataFrame): Input features with columns: distinct_cyclists, popularity_rating, avg_speed, has_bike_lane.
    
    Returns:
        list: Predicted risk levels.
    
    Raises:
        ValueError: If input DataFrame is missing required columns or has invalid data.
    """
    try:
        # Validate input
        required_columns = ['distinct_cyclists', 'popularity_rating', 'avg_speed', 'has_bike_lane']
        missing_cols = [col for col in required_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input DataFrame: {', '.join(missing_cols)}")
        
        # Ensure data types
        X = X.copy()
        X['distinct_cyclists'] = X['distinct_cyclists'].astype(int)
        X['popularity_rating'] = X['popularity_rating'].astype(float)
        X['avg_speed'] = X['avg_speed'].astype(float)
        X['has_bike_lane'] = X['has_bike_lane'].astype(int)
        
        # Predict using DataFrame directly (avoid DMatrix unless necessary)
        predictions = model.predict(X)
        
        # Convert to list for consistency
        return predictions.tolist()
    
    except Exception as e:
        st.error(f"Error in risk prediction: {str(e)}")
        raise
