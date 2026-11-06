"""Task 2 figures."""
import matplotlib.pyplot as plt
import shap

from ..setup import config
from ..utils.plotting import save_fig
from .evaluation import NAIVE

FOLDER = config.FIG_TASK2_DIR


def plot_time_window(weights):
    fig = plt.figure(figsize=(6, 4))
    plt.bar(["Day t-1\n(calendar day)", "Day t\n(calendar day)"], weights["Weight"], color=["tab:grey", "tab:blue"])
    plt.ylabel("Weight in explaining reported AQI (day t)")
    plt.title("Which day do the pollutants behind AQI(t) come from?")
    return save_fig("04_aqi_time_window_check", fig, config.FIG_DIAG_DIR)


def plot_rmse_comparison(results):
    fr = results.copy()
    fr["Label"] = fr.apply(lambda r: r["Model"] if r["Setup"] == "Both" else f"{r['Model']}\n({r['Setup']})", axis=1)
    colors = fr["Setup"].map({"Both": "tab:grey", "Strict": "tab:blue", "Same-day": "tab:orange"})
    naive_rmse = fr.loc[fr["Model"].str.startswith("Naive"), "RMSE"].iloc[0]
    err_lo = (fr["RMSE"] - naive_rmse * (1 + fr["95% CI low (%)"] / 100)).clip(lower=0)
    err_hi = (naive_rmse * (1 + fr["95% CI high (%)"] / 100) - fr["RMSE"]).clip(lower=0)
    fig = plt.figure(figsize=(11, 5))
    plt.bar(fr["Label"], fr["RMSE"], color=colors, yerr=[err_lo, err_hi], capsize=4)
    plt.axhline(naive_rmse, color="black", linestyle="--", lw=1, label="Naive baseline")
    plt.ylabel("RMSE (AQI units)")
    plt.title("Next-day forecast error, 2024 (grey: baselines, blue: strict setup, orange: same-day setup)")
    plt.legend(); plt.xticks(fontsize=8)
    return save_fig("01_nextday_rmse_comparison", fig, FOLDER)


def plot_forecast_timeline(fd, preds, arima_name):
    fig, ax = plt.subplots(figsize=(16, 5))
    dates = fd.target_date[fd.test]
    ax.plot(dates, fd.target[fd.test].values, color="blue", lw=1.2, label="Actual AQI")
    ax.plot(dates, preds[("Both", arima_name)], color="purple", lw=1, label=f"{arima_name} forecast")
    ax.plot(dates, preds[("Strict", "Random Forest")], color="orange", lw=1, label="Random Forest forecast (strict)")
    ax.plot(dates, preds[NAIVE], color="green", lw=0.8, linestyle=":", label="Naive (tomorrow = today)")
    ax.set_title("Next-day AQI forecasts, 2024 (test year)"); ax.set_ylabel("AQI"); ax.legend(ncol=2); ax.grid(True)
    return save_fig("02_nextday_forecast_2024", fig, FOLDER)


def plot_shap_strict(values, X_test):
    plt.figure()
    shap.summary_plot(values, X_test, show=False)
    plt.title("SHAP Summary - Next-day forecast (XGBoost, strict setup, 2024)")
    return save_fig("03_shap_nextday_strict", folder=FOLDER)


def plot_thirty_day(exp):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    history, test = exp["series"]["2024-11-01":"2024-12-01"], exp["test"]
    ax[0].plot(history.index, history, color="grey", label="Training data")
    ax[0].plot(test.index, test, color="blue", marker="o", label="Actual AQI")
    ax[0].plot(test.index, exp["arima_30"], color="orange", linestyle="--", label="ARIMA 30-day forecast")
    ax[0].plot(test.index, exp["naive_30"], color="green", linestyle=":", label="Naive (last value)")
    ax[0].set_title("30-day forecast vs actual (Dec 2024)"); ax[0].set_ylabel("AQI"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(test.index, test, color="blue", marker="o", label="Actual AQI")
    ax[1].plot(test.index, exp["arima_1day"], color="orange", linestyle="--", marker="s", label="ARIMA 1-day-ahead")
    ax[1].plot(test.index, exp["naive_1day"], color="green", linestyle=":", marker="^", label="Naive (tomorrow = today)")
    ax[1].set_title("1-day-ahead forecast vs actual (Dec 2024)"); ax[1].legend(); ax[1].grid(True)
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    return save_fig("04_arima_30day_evaluation", fig, FOLDER)
