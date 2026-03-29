import pandas as pd
from .config import *

def load_base_data():
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    return df

def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

def load_volatility():
    return pd.read_csv(VOL_PATH, parse_dates=["date"])

def load_macro():
    return pd.read_csv(MACRO_PATH, parse_dates=["date"])
