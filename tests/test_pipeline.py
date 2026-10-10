"""
Unit and Integration Tests for CityCycle Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import load_and_preprocess_data, get_feature_columns, get_chronological_split
from models import NaivePersistence24, TrapDemonstrators
from evaluate import compute_metrics

def test_data_integrity_and_anti_leakage():
    df = load_and_preprocess_data("data/hour.csv")
    assert len(df) == 17544, f"Expected 17544 hours, got {len(df)}"
    assert not df.isnull().values.any(), "Preprocessed dataset should have 0 missing values"

    feature_dict = get_feature_columns()
    direct_features = feature_dict["direct"]

    # Target leakage check: ensure target columns are STRICTLY absent from direct feature matrix
    for leaked_col in ["casual", "registered", "cnt", "instant"]:
        assert leaked_col not in direct_features, f"FATAL: Leaked column '{leaked_col}' found in feature matrix!"

def test_chronological_split_strictness():
    df = load_and_preprocess_data("data/hour.csv")
    train_df, test_df = get_chronological_split(df, split_date="2012-10-01")

    assert train_df.index.max() < test_df.index.min(), "Train timestamps must be strictly earlier than test timestamps"
    assert len(test_df) == 2208, f"Expected 2208 test hours in Q4 2012, got {len(test_df)}"

def test_trap_1_target_leakage_demonstration():
    df = load_and_preprocess_data("data/hour.csv")
    res = TrapDemonstrators.run_trap_1_target_leakage(df)
    assert res["r2"] == 1.0, "Trap 1 with casual and registered should yield R2 = 1.0"
    assert res["mae"] < 1e-4, "Trap 1 with casual and registered should yield MAE approx 0.0"

def test_metrics_calculation():
    y_true = np.array([100, 200, 300, 400])
    y_pred = np.array([110, 190, 310, 390])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == 10.0
    assert m["rmse"] == 10.0
    assert m["wape_pct"] == 4.0

def test_exported_artifacts_and_metrics_superiority():
    import json
    artifact_path = "artifacts/dashboard_data.json"
    assert os.path.exists(artifact_path), "Artifact dashboard_data.json must exist"
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
    
    leaderboard = data["benchmarks"]["leaderboard"]
    dual = [m for m in leaderboard if "Dual-Head" in m["model"]][0]
    naive = [m for m in leaderboard if "t-24" in m["model"]][0]

    # Error reduction must be > 35%
    pct_reduction = ((naive["mae"] - dual["mae"]) / naive["mae"]) * 100.0
    assert pct_reduction >= 35.0, f"Expected >= 35% MAE reduction, got {pct_reduction}%"
    assert dual["r2"] > 0.85, f"Expected R2 > 0.85, got {dual['r2']}"

