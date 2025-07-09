import xgboost as xgb
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'xgb_model.pkl')

def load_xgb_model():
    """
    Loads the pre-trained XGBoost model from disk.
    Returns:
        model: Loaded XGBoost model.
    """
    model = joblib.load(MODEL_PATH)
    return model

def predict_risk(model, X):
    """
    Uses the loaded XGBoost model to make predictions.
    Args:
        model: Loaded XGBoost model.
        X (pd.DataFrame): Feature dataframe.
    Returns:
        np.ndarray: Model predictions.
    """
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    return preds
