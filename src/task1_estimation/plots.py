"""Task 1 figures."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap

from ..setup import config
from ..utils.plotting import save_fig

FOLDER = config.FIG_TASK1_DIR


def plot_correlation_heatmap(corr):
    fig = plt.figure(figsize=(12, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 7}, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap - Air Quality Parameters (Training data 2020-2023, measured AQI days)")
    return save_fig("01_correlation_heatmap", fig, FOLDER)


def plot_actual_vs_predicted(split, pred):
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(split.Y_test, pred, color="orange", alpha=0.6)
    lims = [split.Y_test.min(), split.Y_test.max()]
    plt.plot(lims, lims, "r--", lw=2)
    plt.xlabel("Actual AQI"); plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI (Test year 2024)"); plt.grid(True)
    return save_fig("02_actual_vs_predicted_2024", fig, FOLDER)




def plot_monthly_test(predictions, split):
    test = predictions[split.test_rows]
    monthly = test.groupby(test["Timestamp"].dt.month)[["AQI", "Predicted_AQI_RF"]].mean()
    fig = plt.figure(figsize=(11, 6))
    plt.plot(monthly.index, monthly["AQI"], label="Actual AQI", marker="o", color="blue")
    plt.plot(monthly.index, monthly["Predicted_AQI_RF"], label="Predicted AQI (RF)", marker="s", color="orange")
    plt.title("Monthly Average AQI (Actual vs Predicted) - Test Year 2024")
    plt.xlabel("Month"); plt.ylabel("Average AQI"); plt.grid(True); plt.legend()
    return save_fig("03_monthly_actual_vs_predicted_2024", fig, FOLDER)




def plot_shap_summary(values, X_test):
    plt.figure()
    shap.summary_plot(values, X_test, show=False)
    plt.title("SHAP Summary - Random Forest (Test Year 2024)")
    return save_fig("04_shap_summary", folder=FOLDER)


def plot_shap_dependence(values, X_test, feature=config.PM10):
    shap.dependence_plot(feature, values, X_test, show=False)
    plt.title("Effect of PM10 on predicted AQI")
    return save_fig("05_shap_pm10_dependence", folder=FOLDER)


def plot_shap_waterfall(explainer, values, split, prep, model):
    """Explain the prediction for the most polluted measured day of 2024."""
    i = int(split.Y_test.values.argmax())
    day = prep.data.loc[split.X_test.index[i], "Timestamp"].date()
    base = explainer.expected_value[0] if np.ndim(explainer.expected_value) else explainer.expected_value
    plt.figure()
    shap.plots.waterfall(shap.Explanation(values=values[i], base_values=base, data=split.X_test.iloc[i].values,
                                          feature_names=list(split.X_test.columns)), show=False)
    plt.title(f"Why the model predicted this AQI on {day}")
    info = {"date": day, "actual": float(split.Y_test.iloc[i]),
            "predicted": float(model.predict(split.X_test.iloc[[i]])[0])}
    return save_fig("06_shap_waterfall_most_polluted_day", folder=FOLDER), info
