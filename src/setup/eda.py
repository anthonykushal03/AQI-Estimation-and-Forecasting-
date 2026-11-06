"""Exploratory analysis of AQI trends."""
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from ..utils.plotting import save_fig


def yearly_summary(prep):
    """Mean AQI per year, using measured AQI days only."""
    d = prep.data[prep.aqi_measured]
    out = d.groupby(d["Timestamp"].dt.year)["AQI"].agg(["mean", "count"]).reset_index()
    out.columns = ["Year", "Mean AQI (measured days)", "Measured days"]
    return out


def monthly_summary(prep):
    d = prep.data[prep.aqi_measured]
    out = d.groupby(d["Timestamp"].dt.month)["AQI"].mean().reset_index()
    out.columns = ["Month", "Mean AQI (measured days)"]
    return out


def plot_daily_trend(data):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data["Timestamp"], data["AQI"], label="Daily AQI")
    plt.title("Daily AQI Trend (2020–2024)")
    plt.xlabel("Date"); plt.ylabel("AQI")
    plt.grid(True, linestyle="--"); plt.legend()
    return save_fig("01_daily_aqi_trend", fig, config.FIG_DIAG_DIR)


def plot_monthly_trend(data):
    fig = plt.figure()
    data.groupby(data["Timestamp"].dt.month)["AQI"].mean().plot(kind="line", color="orange", marker="o")
    plt.title("Monthly Average AQI Trend (2020-2024)")
    plt.xlabel("Month"); plt.ylabel("Mean AQI")
    return save_fig("02_monthly_aqi_trend", fig, config.FIG_DIAG_DIR)


def plot_wind_direction(data):
    fig = plt.figure(figsize=(9, 4))
    plt.hist(data[config.WD], bins=36, range=(0, 360), color="steelblue", edgecolor="white")
    plt.axvspan(285, 345, color="red", alpha=0.15, label="North-west (285-345°)")
    plt.title("Distribution of daily wind direction (2020-2024)")
    plt.xlabel("Wind direction (degrees)"); plt.ylabel("Number of days"); plt.legend()
    return save_fig("03_wind_direction_check", fig, config.FIG_DIAG_DIR)
