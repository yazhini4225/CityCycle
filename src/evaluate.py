"""
CityCycle Model Evaluation & Benchmark Metrics
==============================================
Calculates honest statistical and operational error metrics:
MAE, RMSE, WAPE (Weighted Absolute Percentage Error), R2, and
stratified peak-hour errors for morning commute, evening return, and weekend surges.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # WAPE (Weighted Absolute Percentage Error): robust to zero/low counts
    total_true = np.sum(y_true)
    wape = float(np.sum(abs_errors) / total_true * 100.0) if total_true > 0 else 0.0

    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "wape_pct": round(wape, 2),
        "r2": round(r2, 4)
    }

def compute_stratified_ops_metrics(df_test: pd.DataFrame, y_true_col: str, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes operational metrics specifically for critical operational time windows:
    - Morning Commuter Rush (Workdays 07:00 - 09:00)
    - Evening Commuter Rush (Workdays 16:00 - 19:00)
    - Weekend Leisure Surge (Weekends 11:00 - 16:00)
    - Off-Peak / Night (All days 00:00 - 06:00)
    """
    y_true = df_test[y_true_col].values
    results = {}

    segments = {
        "morning_commute": (df_test['workingday'] == 1) & (df_test['hr'].between(7, 9)),
        "evening_commute": (df_test['workingday'] == 1) & (df_test['hr'].between(16, 19)),
        "weekend_surge": (df_test['workingday'] == 0) & (df_test['hr'].between(11, 16)),
        "night_offpeak": df_test['hr'].between(0, 6)
    }

    for name, mask in segments.items():
        if mask.sum() > 0:
            m = compute_metrics(y_true[mask], y_pred[mask])
            m["sample_count"] = int(mask.sum())
            m["mean_demand"] = round(float(np.mean(y_true[mask])), 1)
            results[name] = m

    return results
