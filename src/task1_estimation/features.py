"""Task 1 feature selection and train/test preparation. All decisions use training years only."""
from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from ..setup import config
from ..utils import metrics


def training_measured(prep):
    """Training days (2020-2023) where AQI was actually measured."""
    return prep.data[prep.train_mask & prep.aqi_measured]


def correlation_screen(prep):
    """Correlation of every variable with AQI on measured training days, and the pollutants above the threshold."""
    corr = training_measured(prep)[config.ALL_FEATURES + ["AQI"]].corr()
    corr_with_aqi = corr["AQI"].drop("AQI").sort_values(ascending=False)
    selected = [p for p in corr_with_aqi.index if p in config.POLLUTANTS and corr_with_aqi[p] > config.CORR_THRESHOLD]
    return corr, corr_with_aqi, selected


def pca_loadings(prep):
    """PCA on standardised pollutant and weather variables (training days; AQI not included)."""
    x = prep.data[prep.train_mask][config.ALL_FEATURES]
    x = (x - x.mean()) / x.std()
    pca = PCA(n_components=config.N_PCA_COMPONENTS).fit(x)
    loadings = pd.DataFrame(pca.components_.T, index=config.ALL_FEATURES,
                            columns=[f"PC{i + 1}" for i in range(config.N_PCA_COMPONENTS)])
    variance = pd.DataFrame({"Component": loadings.columns, "Explained variance ratio": pca.explained_variance_ratio_})
    return variance, loadings


def toluene_validation(prep):
    """Does Toluene help? Train 2020-2022, validate 2023 (measured AQI days). The 2024 test year is not used."""
    data, years = prep.data, prep.data["Timestamp"].dt.year
    fit_mask = (years < config.VALIDATION_YEAR) & prep.aqi_measured
    val_mask = (years == config.VALIDATION_YEAR) & prep.aqi_measured
    rows = []
    for label, feats in [("With Toluene", config.TASK1_FEATURES + [config.TOLUENE]), ("Without Toluene", config.TASK1_FEATURES)]:
        for name, model in [("Random Forest", RandomForestRegressor(**config.RF_PARAMS)),
                            ("XGBoost", XGBRegressor(**config.XGB_PARAMS))]:
            model.fit(data.loc[fit_mask, feats], data.loc[fit_mask, "AQI"])
            pred = model.predict(data.loc[val_mask, feats])
            rows.append({"Features": label, "Model": name,
                         "Validation R2": metrics.r2(data.loc[val_mask, "AQI"], pred),
                         "Validation RMSE": metrics.rmse(data.loc[val_mask, "AQI"], pred)})
    return pd.DataFrame(rows)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    Y_train: pd.Series
    Y_test: pd.Series
    scaler: MinMaxScaler
    train_rows: pd.Series
    test_rows: pd.Series
    features: list


def make_split(prep, features=None):
    """Temporal split on measured-AQI days, with MinMax scaling fitted on the training data only."""
    features = features or config.TASK1_FEATURES
    X, Y = prep.data[features], prep.data["AQI"]
    train_rows, test_rows = prep.train_mask & prep.aqi_measured, prep.test_mask & prep.aqi_measured
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X[train_rows]), columns=features, index=X[train_rows].index)
    X_test = pd.DataFrame(scaler.transform(X[test_rows]), columns=features, index=X[test_rows].index)
    return Split(X_train, X_test, Y[train_rows], Y[test_rows], scaler, train_rows, test_rows, features)
