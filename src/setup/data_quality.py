"""Data-quality checks: missing-value gaps, split boundaries and wind direction."""
import pandas as pd

from . import config


def missing_aqi_summary(prep):
    """Number of days without measured AQI in each period."""
    return pd.DataFrame({
        "Period": ["Training (2020-2023)", "Test (2024)"],
        "Missing AQI days": [int((~prep.aqi_measured & prep.train_mask).sum()),
                             int((~prep.aqi_measured & prep.test_mask).sum())]})


def boundary_check(raw_by_date, boundaries=("2024-01-01", "2024-12-02")):
    """Missing values on the two days around each boundary. Zero means interpolation cannot
    carry information across that boundary."""
    rows = []
    for b in boundaries:
        b = pd.Timestamp(b)
        window = raw_by_date.loc[b - pd.Timedelta(days=1): b]
        rows.append({"Boundary": b.date(), "Missing values on the two days around it": int(window.isnull().sum().sum())})
    return pd.DataFrame(rows)


def longest_gap(series):
    """Length, start and end of the longest run of consecutive missing days."""
    missing = series.isna()
    if not missing.any():
        return 0, None, None
    runs = series[missing].groupby((missing != missing.shift()).cumsum()[missing])
    longest = max(runs, key=lambda g: len(g[1]))[1]
    return len(longest), longest.index.min().date(), longest.index.max().date()


def gap_table(raw_by_date, columns):
    rows = [(c, int(raw_by_date[c].isna().sum()), *longest_gap(raw_by_date[c])) for c in columns]
    return pd.DataFrame(rows, columns=["Variable", "Missing days", "Longest gap (days)", "Gap starts", "Gap ends"])


def gap_correlation_effect(raw_by_date, train_measured):
    """Correlation with AQI on measured values vs after interpolation (training years, measured AQI)."""
    tr = raw_by_date[(raw_by_date.index.year <= config.TRAIN_END_YEAR) & raw_by_date["AQI"].notna()]
    return pd.DataFrame({
        "Variable": [config.TOLUENE, config.TOLUENE, config.BENZENE, config.BENZENE],
        "Values used": ["measured only", "after interpolation"] * 2,
        "Correlation with AQI": [tr[config.TOLUENE].corr(tr["AQI"]),
                                 train_measured[config.TOLUENE].corr(train_measured["AQI"]),
                                 tr[config.BENZENE].corr(tr["AQI"]),
                                 train_measured[config.BENZENE].corr(train_measured["AQI"])]})


def northwest_wind_share(data):
    """Share of days with daily wind direction in the north-west sector (285-345 degrees)."""
    return data[config.WD].between(285, 345).mean()
