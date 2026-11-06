"""Tests that next-day features never use future rows (src/task2_forecasting/features.py)."""
import numpy as np
import pandas as pd

from src.setup import config
from src.task2_forecasting.features import build_features


def synthetic_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({c: rng.uniform(10, 300, n) for c in ["AQI"] + config.FC_INPUTS}, index=idx)
    df.iloc[[5, 6, 20, 41], 0] = np.nan          # gaps, to exercise forward-filling
    return df


def test_features_unchanged_when_future_removed():
    df = synthetic_data()
    for lag in config.SETUPS.values():
        full = build_features(df, lag)
        for t in df.index[10:]:
            truncated = build_features(df.loc[:t], lag).loc[t]
            assert np.allclose(truncated.values.astype(float), full.loc[t].values.astype(float), equal_nan=True)


def test_strict_setup_uses_yesterdays_pollutants():
    df = synthetic_data()
    F = build_features(df, input_lag=1)
    t = df.index[30]
    assert F.loc[t, f"{config.PM10} yesterday"] == df.loc[df.index[29], config.PM10]


def test_gaps_filled_from_earlier_values_only():
    df = synthetic_data()
    F = build_features(df, input_lag=1)
    assert F.loc[df.index[6], "AQI_today"] == df.loc[df.index[4], "AQI"]
