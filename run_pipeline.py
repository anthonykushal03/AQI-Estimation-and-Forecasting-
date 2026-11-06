"""Run the full project: setup -> Task 1 (estimation) -> Task 2 (forecasting).

    python run_pipeline.py

Outputs (all regenerated on each run):
    results/metrics/          headline results of both tasks
    results/explainability/   SHAP importance tables
    results/diagnostics/      data-quality, feature-selection, timing and leakage checks
    results/predictions/      day-by-day estimated AQI
    results/figures/          figures, in task1_estimation/, task2_forecasting/ and diagnostics/
"""
import matplotlib

matplotlib.use("Agg")

from src.setup import config, data_quality, eda
from src.setup.data import load_and_prepare
from src.task1_estimation import evaluation as t1_eval, features as t1_feat, models as t1_models, plots as t1_plots
from src.task2_forecasting import evaluation as t2_eval, features as t2_feat, models as t2_models, plots as t2_plots
from src.utils.io import save_table

DIAG, EXPLAIN = config.DIAG_DIR, config.EXPLAIN_DIR


def run_setup():
    print("=== Setup: data loading, preprocessing and data-quality checks")
    prep = load_and_prepare()
    print(data_quality.missing_aqi_summary(prep).to_string(index=False))
    save_table(data_quality.boundary_check(prep.raw_by_date), "boundary_missing_check", folder=DIAG)
    save_table(eda.yearly_summary(prep), "yearly_aqi_summary", decimals=2, folder=DIAG)
    save_table(eda.monthly_summary(prep), "monthly_aqi_summary", decimals=2, folder=DIAG)
    eda.plot_daily_trend(prep.data)
    eda.plot_monthly_trend(prep.data)
    eda.plot_wind_direction(prep.data)
    print(f"North-west wind days: {data_quality.northwest_wind_share(prep.data):.1%} (wind direction excluded)")
    return prep


def run_task1(prep):
    print("\n=== Task 1: AQI estimation")
    corr, corr_with_aqi, selected = t1_feat.correlation_screen(prep)
    save_table(corr_with_aqi.rename("Correlation with AQI").reset_index().rename(columns={"index": "Variable"}),
               "correlation_with_aqi", folder=DIAG)
    t1_plots.plot_correlation_heatmap(corr)
    save_table(t1_feat.toluene_validation(prep), "toluene_validation_check", folder=DIAG)
    gaps = data_quality.gap_table(prep.raw_by_date, config.TASK1_FEATURES + [config.TOLUENE, config.BENZENE])
    save_table(gaps, "data_gaps", folder=DIAG)
    save_table(data_quality.gap_correlation_effect(prep.raw_by_date, t1_feat.training_measured(prep)),
               "gap_correlation_effect", folder=DIAG)
    print("Pollutants with r > 0.3:", selected, "-> Toluene excluded; features:", config.TASK1_FEATURES)

    split = t1_feat.make_split(prep)
    models = t1_models.train_models(split)
    results = t1_eval.model_metrics(models, split)
    save_table(results, "task1_estimation_metrics")
    print(results.round(3).to_string(index=False))

    rf = models["RandomForest"]
    predictions = t1_eval.predictions_table(prep, rf, split)
    save_table(predictions, "Predicted_AQI_RF", decimals=None, folder=config.PRED_DIR)
    accuracy = t1_eval.category_accuracy(predictions, split)
    save_table(accuracy, "task1_category_accuracy")
    print(f"Category accuracy: {accuracy['Value'].iloc[0]:.1%}")

    explainer, shap_values, shap_importance = t1_eval.shap_analysis(rf, split)
    save_table(shap_importance, "task1_shap_importance", decimals=3, folder=EXPLAIN)

    t1_plots.plot_actual_vs_predicted(split, rf.predict(split.X_test))
    t1_plots.plot_monthly_test(predictions, split)
    t1_plots.plot_shap_summary(shap_values, split.X_test)
    t1_plots.plot_shap_dependence(shap_values, split.X_test)
    t1_plots.plot_shap_waterfall(explainer, shap_values, split, prep, rf)


def run_task2(prep):
    print("\n=== Task 2: next-day forecasting")
    weights = t2_feat.time_window_check(prep.raw_by_date)
    save_table(weights, "aqi_time_window_check", folder=DIAG)
    t2_plots.plot_time_window(weights)
    w = weights["Weight"].values
    print(f"Time-window check: {w[0] / w.sum():.0%} of the weight on the previous calendar day")

    fd = t2_feat.make_datasets(prep.raw_by_date)
    leak = t2_feat.leakage_test(prep.raw_by_date, fd)
    save_table(leak, "leakage_test", folder=DIAG)
    print(leak.to_string(index=False))

    preds, ml_models, order = t2_models.next_day_predictions(prep.raw_by_date, fd)
    arima_name = f"ARIMA{order}"
    next_day = t2_eval.forecast_metrics(fd, preds)
    save_table(t2_eval.mae_by_season(fd, preds, arima_name), "task2_seasonal_metrics", decimals=2, index=True)
    shap_table, strict_values = t2_eval.shap_by_setup(fd, ml_models)
    save_table(shap_table, "task2_shap_importance", decimals=3, index=True, folder=EXPLAIN)

    t2_plots.plot_rmse_comparison(next_day)
    t2_plots.plot_forecast_timeline(fd, preds, arima_name)
    t2_plots.plot_shap_strict(strict_values, fd.features["Strict"][fd.test])

    print("\n=== Additional experiment: 30-day ARIMA")
    exp = t2_models.thirty_day_experiment(prep)
    thirty = t2_eval.thirty_day_metrics(exp)
    t2_plots.plot_thirty_day(exp)

    results = t2_eval.combined_metrics(next_day, thirty, int(fd.test.sum()))
    save_table(results, "task2_forecasting_metrics")
    print(results[["Experiment", "Setup", "Model", "RMSE", "MAE", "R2", "RMSE change vs naive (%)"]].round(3).to_string(index=False))


def main():
    prep = run_setup()
    run_task1(prep)
    run_task2(prep)
    print(f"\nDone. All outputs are in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
