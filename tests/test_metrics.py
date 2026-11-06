"""Tests for the shared metric functions (src/utils/metrics.py)."""
import numpy as np
import pytest

from src.utils import metrics


def test_categorize_cpcb_boundaries():
    assert metrics.categorize(50) == "Good"
    assert metrics.categorize(51) == "Satisfactory"
    assert metrics.categorize(200) == "Moderate"
    assert metrics.categorize(300) == "Poor"
    assert metrics.categorize(400) == "Very Poor"
    assert metrics.categorize(401) == "Severe"


def test_regression_metrics_known_values():
    m = metrics.regression_metrics([100, 200, 300], [110, 190, 300])
    assert m["MAE"] == pytest.approx(20 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(200 / 3))


def test_category_accuracy():
    assert metrics.category_accuracy([45, 150, 450], [40, 250, 420]) == pytest.approx(2 / 3)


def test_rmse_change_is_zero_against_itself():
    y = np.array([100.0, 200, 300, 250])
    p = np.array([110.0, 190, 280, 260])
    change, lo, hi = metrics.rmse_change_vs_baseline(y, p, p, n_boot=200)
    assert change == pytest.approx(0) and lo == pytest.approx(0) and hi == pytest.approx(0)


def test_rmse_change_negative_when_better():
    y = np.array([100.0, 200, 300, 250, 150])
    change, _, _ = metrics.rmse_change_vs_baseline(y, y + 5, y + 10, n_boot=200)
    assert change == pytest.approx(-50)
