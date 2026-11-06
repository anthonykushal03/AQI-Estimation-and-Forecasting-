"""Layer 1: metric calculation functions, shared by both tasks.

These functions only compute numbers. Task-specific evaluation (which models, which rows,
which baselines) lives in each task's evaluation.py, and the CSV outputs in results/metrics/.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def regression_metrics(y_true, y_pred):
    """RMSE, MAE and R² as a dictionary."""
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred)}


def categorize(aqi):
    """CPCB AQI category."""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"


def category_accuracy(y_true, y_pred):
    """Share of days where the predicted CPCB category matches the actual one."""
    return float(np.mean([categorize(a) == categorize(p) for a, p in zip(y_true, y_pred)]))


def rmse_change_vs_baseline(y_true, y_pred, y_baseline, n_boot=2000, seed=42):
    """Percentage change in RMSE against a baseline (negative = better), with a bootstrap 95% CI.

    The same resampled days are used for the model and the baseline, and the same seed gives
    the same resamples for every model, so the intervals are directly comparable."""
    y_true, y_pred, y_baseline = (np.asarray(a, dtype=float) for a in (y_true, y_pred, y_baseline))
    sq_model, sq_base = (y_true - y_pred) ** 2, (y_true - y_baseline) ** 2
    change = 100 * (np.sqrt(sq_model.mean() / sq_base.mean()) - 1)
    idx = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    boot = 100 * (np.sqrt(sq_model[idx].mean(axis=1) / sq_base[idx].mean(axis=1)) - 1)
    return change, np.percentile(boot, 2.5), np.percentile(boot, 97.5)
