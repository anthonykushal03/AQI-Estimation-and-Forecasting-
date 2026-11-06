"""Task 2 features: time-window check, next-day feature construction and leakage test."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..setup import config


# ---------------------------------------------------------------- time-window check
def sub_index(conc, breakpoints):
    """CPCB sub-index for a pollutant concentration series."""
    out = pd.Series(np.nan, index=conc.index)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        m = (conc >= c_lo) & (conc <= c_hi)
        out[m] = i_lo + (conc[m] - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    out[conc > breakpoints[-1][1]] = 500
    return out


def time_window_check(raw_by_date):
    """How much of the reported AQI on day t is explained by the PM-based AQI of day t-1 vs day t
    (calendar-day averages). A 4 PM-to-4 PM AQI window would put about a third of the weight on day t-1."""
    pm_aqi = np.fmax(sub_index(raw_by_date[config.PM10], config.PM10_BREAKPOINTS),
                     sub_index(raw_by_date[config.PM25], config.PM25_BREAKPOINTS))
    align = pd.DataFrame({"reported": raw_by_date["AQI"], "pm_aqi_yesterday": pm_aqi.shift(1), "pm_aqi_today": pm_aqi}).dropna()
    w = LinearRegression().fit(align[["pm_aqi_yesterday", "pm_aqi_today"]], align["reported"]).coef_
    return pd.DataFrame({"Term": ["PM-based AQI, day t-1", "PM-based AQI, day t"], "Weight": w})


# ---------------------------------------------------------------- next-day features
def build_features(df, input_lag):
    """Features for day t. AQI history up to day t; pollutants and weather from day t - input_lag
    (1 = strict setup, 0 = same-day setup). Gaps are forward-filled, so only earlier values are used."""
    past = df.ffill()
    F = pd.DataFrame(index=df.index)
    F["AQI_today"] = past["AQI"]
    for lag in range(1, 7):
        F[f"AQI_{lag}d_ago"] = past["AQI"].shift(lag)
    F["AQI_mean_7d"] = past["AQI"].rolling(7).mean()
    when = "yesterday" if input_lag == 1 else "today"
    for c in config.FC_INPUTS:
        F[f"{c} {when}"] = past[c].shift(input_lag)
    F["Month"] = F.index.month
    return F


@dataclass
class ForecastData:
    features: dict          # setup name -> feature table
    target: pd.Series       # tomorrow's measured AQI (missing if not measured)
    target_date: pd.DatetimeIndex
    train: pd.Series        # rows whose TARGET date is in 2020-2023
    test: pd.Series         # rows whose TARGET date is in 2024


def make_datasets(raw_by_date):
    """Both setups on exactly the same rows; split by target date; measured targets only."""
    features = {s: build_features(raw_by_date, lag) for s, lag in config.SETUPS.items()}
    target = raw_by_date["AQI"].shift(-1)
    target_date = raw_by_date.index + pd.Timedelta(days=1)
    valid = target.notna()
    for F in features.values():
        valid &= F.notna().all(axis=1)
    train = valid & (target_date.year <= config.TRAIN_END_YEAR)
    test = valid & (target_date.year == config.TEST_YEAR)
    return ForecastData(features, target, target_date, train, test)


def leakage_test(raw_by_date, fd, n_days=config.LEAKAGE_TEST_DAYS, seed=0):
    """Rebuild the features for random test days using only data up to that day. Identical features mean
    no future ROWS are used. (This cannot detect overlap inside a day's averaging window.)"""
    days = np.random.default_rng(seed).choice(fd.features["Strict"].index[fd.test], size=n_days, replace=False)
    rows = []
    for setup, lag in config.SETUPS.items():
        passed = sum(np.allclose(build_features(raw_by_date.loc[:t], lag).loc[t].values.astype(float),
                                 fd.features[setup].loc[t].values.astype(float), equal_nan=True) for t in days)
        rows.append({"Setup": setup, "Days checked": n_days, "Days identical": int(passed)})
    result = pd.DataFrame(rows)
    split_ok = fd.target_date[fd.train].max() < pd.Timestamp(f"{config.TEST_YEAR}-01-01")
    if (result["Days identical"] < n_days).any() or not split_ok:
        raise AssertionError(f"Leakage test failed:\n{result}\nAll training targets before test year: {split_ok}")
    return result
