"""Task 1 models: Linear Regression, Random Forest and XGBoost (settings fixed in advance)."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from ..setup import config


def train_models(split):
    models = {"Linear": LinearRegression(),
              "RandomForest": RandomForestRegressor(**config.RF_PARAMS),
              "XGBoost": XGBRegressor(**config.XGB_PARAMS)}
    for model in models.values():
        model.fit(split.X_train, split.Y_train)
    return models
