"""Layer 2: Task 1 evaluation (which models, which rows), built on the metric functions in utils.metrics."""
import numpy as np
import pandas as pd
import shap

from ..utils import metrics


def model_metrics(models, split):
    """Train R² and test R², RMSE and MAE for each model (test = 2024, measured AQI days)."""
    rows = []
    for name, model in models.items():
        test = metrics.regression_metrics(split.Y_test, model.predict(split.X_test))
        rows.append({"Model": name, "Train R2": metrics.r2(split.Y_train, model.predict(split.X_train)),
                     "Test R2": test["R2"], "Test RMSE": test["RMSE"], "Test MAE": test["MAE"]})
    return pd.DataFrame(rows)


def predictions_table(prep, model, split):
    """Actual and estimated AQI with CPCB categories for every day (no raw pollutant columns)."""
    data = prep.data
    X_all = pd.DataFrame(split.scaler.transform(data[split.features]), columns=split.features, index=data.index)
    out = pd.DataFrame({"Timestamp": data["Timestamp"], "AQI": data["AQI"], "AQI_measured": prep.aqi_measured})
    out["AQI_Class"] = out["AQI"].apply(metrics.categorize)
    out["Predicted_AQI_RF"] = model.predict(X_all)
    out["Predicted_AQI_Class_RF"] = out["Predicted_AQI_RF"].apply(metrics.categorize)
    return out


def category_accuracy(predictions, split):
    test = predictions[split.test_rows]
    acc = metrics.category_accuracy(test["AQI"], test["Predicted_AQI_RF"])
    return pd.DataFrame({"Metric": ["AQI category accuracy (2024, measured AQI days)"], "Value": [round(acc, 4)]})




def shap_analysis(model, split):
    """SHAP values on the 2024 test set and mean |SHAP| per feature (in AQI units)."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(split.X_test)
    importance = pd.DataFrame({"Feature": split.X_test.columns, "Mean |SHAP| (AQI units)": np.abs(values).mean(axis=0)}
                              ).sort_values("Mean |SHAP| (AQI units)", ascending=False)
    return explainer, values, importance
