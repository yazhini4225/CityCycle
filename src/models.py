"""
CityCycle Machine Learning Models & Baselines
=============================================
Contains naive operational baselines, regularized linear models, tree ensembles,
the innovative Dual-Head XGBoost architecture, and diagnostic trap demonstrators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class NaivePersistence24:
    """Predicts exactly what demand was at the same hour yesterday (t-24)."""
    def __init__(self):
        self.name = "Naive Baseline (t-24 Persistence)"

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        return df_test['lag_24'].values

class NaiveWeekly168:
    """Predicts exactly what demand was at the same hour last week (t-168)."""
    def __init__(self):
        self.name = "Naive Baseline (t-168 Weekly Persistence)"

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        return df_test['lag_168'].values

class SeasonalHistoricalMean:
    """
    Predicts the historical training average demand for the specific (weekday, hour) slot.
    A standard benchmark in operations planning.
    """
    def __init__(self):
        self.name = "Seasonal Historical Mean Baseline"
        self.lookup: Dict[Tuple[int, int], float] = {}
        self.global_mean: float = 0.0

    def fit(self, df_train: pd.DataFrame):
        self.global_mean = float(df_train['cnt'].mean())
        grouped = df_train.groupby(['weekday', 'hr'])['cnt'].mean()
        self.lookup = grouped.to_dict()
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df_test.iterrows():
            key = (int(row['weekday']), int(row['hr']))
            preds.append(self.lookup.get(key, self.global_mean))
        return np.array(preds)

class RidgeRegressionModel:
    """L2 Regularized linear model with standardized feature inputs."""
    def __init__(self, alpha: float = 10.0):
        self.name = "Linear Ridge Regression"
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        preds = self.pipeline.predict(X_test)
        return np.clip(preds, 0, None)

class DirectXGBoostModel:
    """Monolithic Gradient Boosted Decision Tree predicting total count directly."""
    def __init__(self, n_estimators: int = 350, max_depth: int = 6, learning_rate: float = 0.04):
        self.name = "Direct XGBoost"
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X_test)
        return np.clip(preds, 0, None)

class DualHeadXGBoostModel:
    """
    Innovative Dual-Head Architecture:
    - Head A models 'registered' demand (rigid commuter pulse, rush hours, workdays).
    - Head B models 'casual' demand (elastic leisure/tourist surge, temperature & weekends).
    - Total demand = Registered Forecast + Casual Forecast.
    Prevents leakage while unlocking sub-segment operational intelligence.
    """
    def __init__(self, n_estimators: int = 350, max_depth: int = 6, learning_rate: float = 0.04):
        self.name = "Dual-Head XGBoost (Registered + Casual)"
        self.model_reg = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        self.model_cas = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, df_train: pd.DataFrame, feat_reg: list, feat_cas: list):
        self.model_reg.fit(df_train[feat_reg], df_train['registered'])
        self.model_cas.fit(df_train[feat_cas], df_train['casual'])
        return self

    def predict_components(self, df_test: pd.DataFrame, feat_reg: list, feat_cas: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_reg = np.clip(self.model_reg.predict(df_test[feat_reg]), 0, None)
        pred_cas = np.clip(self.model_cas.predict(df_test[feat_cas]), 0, None)
        total_pred = pred_reg + pred_cas
        return total_pred, pred_reg, pred_cas

    def predict(self, df_test: pd.DataFrame, feat_reg: list, feat_cas: list) -> np.ndarray:
        total, _, _ = self.predict_components(df_test, feat_reg, feat_cas)
        return total

class TrapDemonstrators:
    """
    Empirical Demonstrators for the two critical traps described in prompt:
    1. Trap 1: Target leakage with casual & registered (fake R2 = 1.0)
    2. Trap 2: Random shuffle cheat vs chronological split
    """
    @staticmethod
    def run_trap_1_target_leakage(df: pd.DataFrame) -> Dict[str, float]:
        """Shows why using casual and registered creates a 100% accurate yet useless model."""
        lr = LinearRegression()
        X_leak = df[['casual', 'registered']]
        y = df['cnt']
        lr.fit(X_leak, y)
        preds = lr.predict(X_leak)
        mae = float(np.mean(np.abs(y - preds)))
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        r2 = 1.0 - (np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2))
        return {
            "mae": float(round(mae, 6)),
            "rmse": float(round(rmse, 6)),
            "r2": float(round(r2, 6)),
            "weights": {"casual_coef": float(lr.coef_[0]), "registered_coef": float(lr.coef_[1]), "intercept": float(lr.intercept_)}
        }
