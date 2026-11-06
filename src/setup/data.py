"""Loading and preprocessing of the CPCB daily data."""
from dataclasses import dataclass

import pandas as pd

from . import config


@dataclass
class PreparedData:
    data: pd.DataFrame                  # sorted; inputs interpolated separately in train and test periods
    raw_by_date: pd.DataFrame           # original values (with gaps), indexed by date
    train_mask: pd.Series               # 2020-2023
    test_mask: pd.Series                # 2024
    aqi_measured: pd.Series             # True where AQI was actually measured (before interpolation)
    aqi_measured_by_date: pd.Series     # same, indexed by date


def load_raw(data_dir=config.DATA_DIR):
    """Read the yearly CSV files and stack them into one table."""
    files = [data_dir / f"Data_{year}(1).csv" for year in config.YEARS]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files in {data_dir}: {missing}. See data/README.md.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def prepare(raw):
    """Sort by date, record measured AQI, and interpolate missing values separately within
    the training and test periods (so no 2024 value fills a 2023 gap)."""
    data = raw.copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.sort_values("Timestamp")

    train_mask = data["Timestamp"].dt.year <= config.TRAIN_END_YEAR
    test_mask = data["Timestamp"].dt.year == config.TEST_YEAR

    aqi_measured = data["AQI"].notna()
    aqi_measured_by_date = pd.Series(aqi_measured.values, index=data["Timestamp"])

    num_cols = data.columns.drop("Timestamp")
    data.loc[train_mask, num_cols] = data.loc[train_mask, num_cols].interpolate(limit_direction="both")
    data.loc[test_mask, num_cols] = data.loc[test_mask, num_cols].interpolate(limit_direction="both")

    raw_by_date = raw.copy()
    raw_by_date["Timestamp"] = pd.to_datetime(raw_by_date["Timestamp"])
    raw_by_date = raw_by_date.set_index("Timestamp").sort_index()

    return PreparedData(data, raw_by_date, train_mask, test_mask, aqi_measured, aqi_measured_by_date)


def load_and_prepare(data_dir=config.DATA_DIR):
    return prepare(load_raw(data_dir))
