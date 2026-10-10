"""
CityCycle Data Loader & Anti-Leakage Feature Pipeline
=====================================================
Ingests UCI Bike Sharing hourly dataset ('hour.csv'), handles time-series reindexing,
strictly prevents target leakage (excludes casual, registered, cnt from feature matrix),
and produces causal temporal features and chronological train/test splits.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

def load_and_preprocess_data(csv_path: str = "data/hour.csv") -> pd.DataFrame:
    """
    Loads hour.csv, reindexes to continuous 1-hour timeline, creates causal calendar,
    cyclical, weather comfort, and retrospective lag features.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    raw_df = pd.read_csv(csv_path)
    
    # Construct exact continuous timestamp
    raw_df['dt'] = pd.to_datetime(raw_df['dteday']) + pd.to_timedelta(raw_df['hr'], unit='h')
    df = raw_df.set_index('dt').sort_index()

    # Reindex over the complete date range (handles missing overnight storm hours)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='1h')
    df = df.reindex(full_idx)

    # Impute missing counts with 0 (system was closed or 0 trips)
    df['cnt'] = df['cnt'].fillna(0)
    df['casual'] = df['casual'].fillna(0)
    df['registered'] = df['registered'].fillna(0)
    df['instant'] = np.arange(1, len(df) + 1)

    # Forward-fill atmospheric variables
    weather_cols = ['temp', 'atemp', 'hum', 'windspeed', 'weathersit', 'season', 'holiday', 'workingday']
    for col in weather_cols:
        df[col] = df[col].ffill().bfill()

    # Calendar & temporal fields
    df['hr'] = df.index.hour
    df['weekday'] = df.index.weekday  # 0=Monday, 6=Sunday
    df['mnth'] = df.index.month
    df['yr'] = df.index.year - 2011  # 0=2011, 1=2012
    df['dteday'] = df.index.strftime('%Y-%m-%d')

    # Cyclical representations
    df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24.0)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24.0)
    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12.0)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12.0)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7.0)

    # Denormalized human-readable weather metrics
    df['temp_c'] = df['temp'] * 41.0                 # Real Celsius
    df['atemp_c'] = df['atemp'] * 50.0               # Feeling Celsius
    df['hum_pct'] = df['hum'] * 100.0                # Percentage
    df['windspeed_kmh'] = df['windspeed'] * 67.0     # km/h
    df['temp_feel_diff'] = df['atemp_c'] - df['temp_c']

    # Operational indicators
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_commute_morning'] = ((df['workingday'] == 1) & (df['hr'].between(7, 9))).astype(int)
    df['is_commute_evening'] = ((df['workingday'] == 1) & (df['hr'].between(16, 19))).astype(int)
    df['is_weekend_midday'] = ((df['workingday'] == 0) & (df['hr'].between(11, 16))).astype(int)
    df['severe_weather'] = (df['weathersit'] >= 3).astype(int)

    # Thermal comfort score: comfort penalised by extreme humidity or cold wind
    df['comfort_index'] = (
        df['temp_c'] 
        - 0.08 * np.maximum(0, df['hum_pct'] - 65.0) 
        - 0.15 * np.maximum(0, df['windspeed_kmh'] - 20.0)
    )

    # Causal lag features (lag >= 24 hours to guarantee day-ahead rebalancing schedule feasibility)
    df['lag_24'] = df['cnt'].shift(24).bfill()
    df['lag_48'] = df['cnt'].shift(48).bfill()
    df['lag_168'] = df['cnt'].shift(168).bfill()  # Same hour exactly 1 week ago
    df['roll_mean_24_lag24'] = df['cnt'].shift(24).rolling(24, min_periods=1).mean().bfill()
    df['roll_std_24_lag24'] = df['cnt'].shift(24).rolling(24, min_periods=1).std().fillna(0).bfill()

    # Sub-component causal lags for Dual-Head modeling
    df['reg_lag_24'] = df['registered'].shift(24).bfill()
    df['reg_lag_168'] = df['registered'].shift(168).bfill()
    df['cas_lag_24'] = df['casual'].shift(24).bfill()
    df['cas_lag_168'] = df['casual'].shift(168).bfill()

    return df

def get_feature_columns() -> Dict[str, List[str]]:
    """Returns the explicit feature columns for direct and dual-head models."""
    base_features = [
        'hr', 'weekday', 'mnth', 'season', 'workingday', 'holiday',
        'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'weekday_sin', 'weekday_cos',
        'temp_c', 'atemp_c', 'hum_pct', 'windspeed_kmh', 'temp_feel_diff',
        'weathersit', 'severe_weather', 'comfort_index',
        'is_weekend', 'is_commute_morning', 'is_commute_evening', 'is_weekend_midday',
        'lag_24', 'lag_48', 'lag_168', 'roll_mean_24_lag24', 'roll_std_24_lag24'
    ]

    reg_features = base_features + ['reg_lag_24', 'reg_lag_168']
    cas_features = base_features + ['cas_lag_24', 'cas_lag_168']

    return {
        "direct": base_features,
        "registered": reg_features,
        "casual": cas_features
    }

def get_chronological_split(df: pd.DataFrame, split_date: str = "2012-10-01") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs an honest chronological time-series split.
    Training/Validation: 2011-01-01 to 2012-09-30 (21 months)
    Held-out Test: 2012-10-01 to 2012-12-31 (Q4 2012, 3 months)
    """
    train_df = df[df.index < split_date].copy()
    test_df = df[df.index >= split_date].copy()
    return train_df, test_df
