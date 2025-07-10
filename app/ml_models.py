import xgboost as xgb
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'xgb_model.pkl')

def load_xgb_model():
    """
    Loads the pre-trained XGBoost Booster model from disk.
    Returns:
        model: Loaded XGBoost Booster model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

def predict_risk(model, X):
    """
    Uses the loaded XGBoost Booster model to make predictions.
    Args:
        model: Loaded XGBoost Booster model.
        X (pd.DataFrame): Feature dataframe.
    Returns:
        np.ndarray: Model predictions.
    """
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    return preds
