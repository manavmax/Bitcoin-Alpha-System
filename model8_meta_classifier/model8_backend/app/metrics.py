import pandas as pd
from .data_loader import *

def compute_metrics(threshold):
    df = load_predictions()

    active = df[df["confidence"] >= threshold]
    coverage = len(active) / len(df)

    da = (active["pred_class"] == active["true_label"]).mean()

    return coverage, da
