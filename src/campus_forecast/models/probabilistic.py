"""Quantile regressors and validation-only conformal interval calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from campus_forecast.exceptions import DataValidationError


class QuantileDirectForecaster:
    """Fit separate direct regressors for P10/P50/P90 and each horizon."""

    def __init__(
        self,
        *,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
        random_state: int = 42,
        n_estimators: int = 100,
    ) -> None:
        if any(not 0 < quantile < 1 for quantile in quantiles):
            raise ValueError("quantiles must lie in (0, 1)")
        self.quantiles = tuple(sorted(quantiles))
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.models: dict[float, list[object]] = {}
        self.horizon_steps: int | None = None

    def _factory(self, quantile: float):
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_state,
                verbosity=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=min(self.n_estimators, 200),
                random_state=self.random_state,
            )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> QuantileDirectForecaster:
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) == 0:
            raise DataValidationError(
                "Quantile training data must be non-empty [samples, features/horizons]"
            )
        self.horizon_steps = y.shape[1]
        self.models = {}
        for quantile in self.quantiles:
            self.models[quantile] = []
            for horizon in range(self.horizon_steps):
                model = self._factory(quantile)
                model.fit(x, y[:, horizon])
                self.models[quantile].append(model)
        return self

    def predict(self, features: np.ndarray) -> dict[float, np.ndarray]:
        if not self.models:
            raise RuntimeError("Quantile forecaster has not been fitted")
        x = np.asarray(features, dtype=float)
        predictions = {
            quantile: np.column_stack([model.predict(x) for model in models])
            for quantile, models in self.models.items()
        }
        # Quantile crossing is repaired per sample/horizon without changing the
        # meaning of the requested quantiles.
        for index in range(x.shape[0]):
            for horizon in range(self.horizon_steps or 0):
                values = np.array([predictions[q][index, horizon] for q in self.quantiles])
                values.sort()
                for quantile, value in zip(self.quantiles, values, strict=True):
                    predictions[quantile][index, horizon] = value
        return predictions


@dataclass
class ConformalIntervalCalibrator:
    """Split-conformal expansion calibrated on a held-out calibration period."""

    coverage_level: float = 0.80
    adjustment_: float | None = None

    def fit(self, actual: object, lower: object, upper: object) -> ConformalIntervalCalibrator:
        if not 0 < self.coverage_level < 1:
            raise ValueError("coverage_level must be between 0 and 1")
        y = np.asarray(actual, dtype=float)
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        if not (y.shape == lo.shape == hi.shape):
            raise ValueError("actual and interval arrays must have equal shape")
        scores = np.maximum(lo - y, y - hi).reshape(-1)
        scores = scores[np.isfinite(scores)]
        if not len(scores):
            raise DataValidationError("No finite calibration scores were available")
        # Conservative finite-sample quantile for split conformal calibration.
        level = min(1.0, np.ceil((len(scores) + 1) * self.coverage_level) / len(scores))
        self.adjustment_ = float(np.quantile(np.maximum(scores, 0), level, method="higher"))
        return self

    def transform(self, lower: object, upper: object) -> tuple[np.ndarray, np.ndarray]:
        if self.adjustment_ is None:
            raise RuntimeError("Calibrator has not been fitted")
        lo = np.asarray(lower, dtype=float) - self.adjustment_
        hi = np.asarray(upper, dtype=float) + self.adjustment_
        return lo, hi
