"""Layer 2: Task 2 evaluation, built on the metric functions in utils.metrics."""
import numpy as np
import pandas as pd
import shap

from ..setup import config
from ..utils import metrics

SEASONS = {12: "Winter (Dec-Feb)", 1: "Winter (Dec-Feb)", 2: "Winter (Dec-Feb)",
           3: "Summer (Mar-Jun)", 4: "Summer (Mar-Jun)", 5: "Summer (Mar-Jun)", 6: "Summer (Mar-Jun)",
           7: "Monsoon (Jul-Sep)", 8: "Monsoon (Jul-Sep)", 9: "Monsoon (Jul-Sep)",
           10: "Post-monsoon (Oct-Nov)", 11: "Post-monsoon (Oct-Nov)"}
NAIVE = ("Both", "Naive (tomorrow = today)")


def forecast_metrics(fd, preds):
    """RMSE, MAE, R², category accuracy and RMSE change vs naive (bootstrap 95% CI) for every model."""
    y = fd.target[fd.test].values
    rows = []
    for (setup, model), p in preds.items():
        change, lo, hi = metrics.rmse_change_vs_baseline(y, p, preds[NAIVE], n_boot=config.N_BOOTSTRAP, seed=config.SEED)
        rows.append({"Setup": setup, "Model": model, **metrics.regression_metrics(y, p),
                     "Category accuracy": metrics.category_accuracy(y, p),
                     "RMSE change vs naive (%)": change, "95% CI low (%)": lo, "95% CI high (%)": hi})
    return pd.DataFrame(rows)


def mae_by_season(fd, preds, arima_name):
    y = fd.target[fd.test].values
    season = pd.Series(fd.target_date[fd.test].month).map(SEASONS).values
    err = pd.DataFrame({"Season": season,
                        "Naive": np.abs(y - preds[NAIVE]),
                        arima_name: np.abs(y - preds[("Both", arima_name)]),
                        "Random Forest (strict)": np.abs(y - preds[("Strict", "Random Forest")]),
                        "XGBoost (strict)": np.abs(y - preds[("Strict", "XGBoost")])})
    return err.groupby("Season").mean()


def shap_by_setup(fd, ml_models):
    """Mean |SHAP| per feature for the XGBoost model of each setup. Returns (table, strict SHAP values)."""
    tables, strict_values = {}, None
    for setup in config.SETUPS:
        X_test = fd.features[setup][fd.test]
        values = shap.TreeExplainer(ml_models[setup]["XGBoost"]).shap_values(X_test)
        tables[setup] = pd.Series(np.abs(values).mean(axis=0).astype(float), index=X_test.columns).sort_values(ascending=False)
        if setup == "Strict":
            strict_values = values
    table = pd.concat(tables, names=["Setup", "Feature"]).rename("Mean |SHAP| (AQI units)")
    return table, strict_values


def thirty_day_metrics(exp):
    """Scores for the 30-day experiment, on measured AQI days only."""
    days = exp["eval_days"]
    actual = exp["test"][days]
    rows = []
    for test_name, model, pred in [("30-day forecast", "ARIMA(30,2,0)", exp["arima_30"]),
                                   ("30-day forecast", "Naive baseline", exp["naive_30"]),
                                   ("1-day-ahead", "ARIMA(30,2,0)", exp["arima_1day"]),
                                   ("1-day-ahead", "Naive baseline", exp["naive_1day"])]:
        rows.append({"Test": test_name, "Model": model, "RMSE": metrics.rmse(actual, pred[days]),
                     "MAE": metrics.mae(actual, pred[days]), "Days evaluated": int(days.sum())})
    return pd.DataFrame(rows)


def combined_metrics(next_day, thirty_day, n_test_days):
    """One Task 2 results table: the next-day forecasts (2024) and the additional 30-day ARIMA experiment
    (Dec 2024). Columns that do not apply to the 30-day experiment are left empty."""
    nd = next_day.copy()
    nd.insert(0, "Experiment", "Next-day forecast (2024)")
    nd["Days evaluated"] = n_test_days
    td = thirty_day.copy()
    td.insert(0, "Experiment", "30-day ARIMA experiment (Dec 2024): " + td.pop("Test"))
    td.insert(1, "Setup", "Both")
    return pd.concat([nd, td], ignore_index=True)[list(nd.columns)]
