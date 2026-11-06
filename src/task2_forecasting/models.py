"""Task 2 models: naive persistence, ARIMA (order by AIC on training years), Random Forest and XGBoost;
plus the ARIMA models for the additional 30-day experiment."""
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from ..setup import config


def _fit_arima(series, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # hide convergence messages
        return ARIMA(series, order=order).fit()


def select_arima_order(train_series):
    """Lowest-AIC ARIMA order on the training period only."""
    best_aic, best_order = float("inf"), None
    g = config.ARIMA_ORDER_GRID
    for p in g["p"]:
        for d in g["d"]:
            for q in g["q"]:
                try:
                    aic = _fit_arima(train_series, (p, d, q)).aic
                    if aic < best_aic:
                        best_aic, best_order = aic, (p, d, q)
                except Exception:
                    pass
    return best_order


def next_day_predictions(raw_by_date, fd):
    """Predictions for every 2024 test row from all models. Returns (predictions, fitted ML models, ARIMA order).
    Prediction keys are (setup, model); baselines use setup 'Both' because they have no pollutant inputs."""
    test_index = fd.features["Strict"].index[fd.test]
    preds = {("Both", "Naive (tomorrow = today)"): fd.features["Strict"].loc[fd.test, "AQI_today"].values}

    aqi_past = raw_by_date["AQI"].ffill().asfreq("D")
    train_series = aqi_past[:f"{config.TRAIN_END_YEAR}-12-31"]
    order = select_arima_order(train_series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima_all = _fit_arima(train_series, order).apply(aqi_past)       # same parameters over the full series
    arima_next = arima_all.predict().shift(-1)                            # value at t = forecast for t+1
    preds[("Both", f"ARIMA{order}")] = arima_next.reindex(test_index).values

    ml_models = {}
    y_train = fd.target[fd.train]
    for setup in config.SETUPS:
        X_train, X_test = fd.features[setup][fd.train], fd.features[setup][fd.test]
        rf = RandomForestRegressor(**config.RF_PARAMS).fit(X_train, y_train)
        xgb = XGBRegressor(**config.XGB_PARAMS).fit(X_train, y_train)
        ml_models[setup] = {"Random Forest": rf, "XGBoost": xgb}
        preds[(setup, "Random Forest")] = rf.predict(X_test)
        preds[(setup, "XGBoost")] = xgb.predict(X_test)
    return preds, ml_models, order


# ---------------------------------------------------------------- additional 30-day experiment
def thirty_day_experiment(prep):
    """ARIMA(30,2,0) trained on everything before the last 30 days, then a 30-day forecast and
    1-day-ahead forecasts (fixed parameters) over the held-out window, with naive baselines."""
    series = prep.data.set_index("Timestamp")["AQI"].asfreq("D")
    train, test = series[:-config.HOLDOUT_DAYS], series[-config.HOLDOUT_DAYS:]
    fit = _fit_arima(train, config.ARIMA_30DAY_ORDER)
    return {
        "series": series, "train": train, "test": test,
        "eval_days": prep.aqi_measured_by_date.loc[test.index].values,
        "arima_30": fit.forecast(steps=config.HOLDOUT_DAYS),
        "naive_30": pd.Series(train.iloc[-1], index=test.index),
        "arima_1day": fit.apply(series).predict(start=test.index[0], end=test.index[-1]),
        "naive_1day": series.shift(1)[-config.HOLDOUT_DAYS:],
    }
