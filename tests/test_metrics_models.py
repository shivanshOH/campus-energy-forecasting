from __future__ import annotations

import numpy as np

from campus_forecast.evaluation.metrics import evaluate_point, evaluate_probabilistic
from campus_forecast.models.baselines import LinearRegressionForecaster
from campus_forecast.models.probabilistic import (
    ConformalIntervalCalibrator,
    QuantileDirectForecaster,
)


def test_point_metrics_use_original_units() -> None:
    metrics = evaluate_point([10, 20], [12, 18])
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert metrics["r2"] == 0.84


def test_probability_metrics_and_conformal_calibration() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 1.0, 2.0, 3.0])
    upper = np.array([2.0, 3.0, 4.0, 5.0])
    metrics = evaluate_probabilistic(actual, lower, actual, upper)
    assert metrics["interval_coverage_80"] == 1.0
    calibrator = ConformalIntervalCalibrator(coverage_level=0.8).fit(actual, lower, upper)
    calibrated_lower, calibrated_upper = calibrator.transform(lower, upper)
    assert np.all(calibrated_lower <= calibrated_upper)


def test_direct_regressor_is_multi_horizon() -> None:
    x = np.arange(30, dtype=float).reshape(15, 2)
    y = np.column_stack([x[:, 0] + 1, x[:, 1] + 2])
    model = LinearRegressionForecaster().fit(x, y)
    prediction = model.predict(x[:3])
    assert prediction.shape == (3, 2)


def test_quantile_direct_model_returns_ordered_intervals() -> None:
    x = np.arange(80, dtype=float).reshape(40, 2)
    y = np.column_stack([x[:, 0] + np.sin(x[:, 0]), x[:, 1] + 2])
    model = QuantileDirectForecaster(n_estimators=10).fit(x, y)
    prediction = model.predict(x[:4])
    assert prediction[0.1].shape == (4, 2)
    assert np.all(prediction[0.1] <= prediction[0.5])
    assert np.all(prediction[0.5] <= prediction[0.9])
