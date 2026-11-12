"""Point forecasting baselines with direct multi-horizon outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from campus_forecast.exceptions import DataValidationError


class DirectForecaster(ABC):
    """Minimal model interface shared by classical and neural models."""

    @abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray) -> DirectForecaster:
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DirectSklearnForecaster(DirectForecaster):
    """Fit one estimator per horizon to avoid recursive error accumulation."""

    estimator_factory: object
    horizon_steps: int | None = None
    estimators: list[object] | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> DirectSklearnForecaster:
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be [samples, features] and targets [samples, horizons]")
        if len(x) == 0:
            raise DataValidationError("Cannot train a model with zero samples")
        self.horizon_steps = y.shape[1]
        self.estimators = []
        for horizon in range(self.horizon_steps):
            estimator = (
                self.estimator_factory()
                if callable(self.estimator_factory)
                else self.estimator_factory
            )
            estimator.fit(x, y[:, horizon])
            self.estimators.append(estimator)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.estimators:
            raise RuntimeError("Forecaster has not been fitted")
        x = np.asarray(features, dtype=float)
        return np.column_stack([estimator.predict(x) for estimator in self.estimators])


class LinearRegressionForecaster(DirectSklearnForecaster):
    def __init__(self, *, alpha: float = 1.0) -> None:
        from sklearn.linear_model import Ridge

        super().__init__(estimator_factory=lambda: Ridge(alpha=alpha))


class DecisionTreeForecaster(DirectSklearnForecaster):
    def __init__(
        self, *, max_depth: int = 14, min_samples_split: int = 20, random_state: int = 42
    ) -> None:
        from sklearn.tree import DecisionTreeRegressor

        super().__init__(
            estimator_factory=lambda: DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
            )
        )


class RandomForestForecaster(DirectSklearnForecaster):
    def __init__(
        self, *, n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1
    ) -> None:
        from sklearn.ensemble import RandomForestRegressor

        super().__init__(
            estimator_factory=lambda: RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
                min_samples_leaf=1,
            )
        )


class HistGradientBoostingForecaster(DirectSklearnForecaster):
    def __init__(self, *, random_state: int = 42) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        super().__init__(
            estimator_factory=lambda: HistGradientBoostingRegressor(random_state=random_state)
        )


class LightGBMForecaster(DirectSklearnForecaster):
    """LightGBM comparison model with an explicit CPU fallback."""

    def __init__(self, *, random_state: int = 42, n_estimators: int = 100) -> None:
        try:
            from lightgbm import LGBMRegressor

            def factory():
                return LGBMRegressor(
                    objective="regression",
                    n_estimators=n_estimators,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=random_state,
                    verbosity=-1,
                )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            def factory():
                return HistGradientBoostingRegressor(random_state=random_state)

        super().__init__(estimator_factory=factory)


class PersistenceForecaster:
    """Repeat the last observed energy value for each requested horizon."""

    def predict_from_history(self, history: np.ndarray, horizon_steps: int) -> np.ndarray:
        values = np.asarray(history, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("history must be a non-empty [samples, time] array")
        return np.repeat(values[:, -1:], horizon_steps, axis=1)


class SeasonalNaiveForecaster:
    """Repeat the value one seasonal cycle earlier when supplied in history."""

    def __init__(self, seasonal_steps: int = 144) -> None:
        if seasonal_steps < 1:
            raise ValueError("seasonal_steps must be positive")
        self.seasonal_steps = seasonal_steps

    def predict_from_history(self, history: np.ndarray, horizon_steps: int) -> np.ndarray:
        values = np.asarray(history, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("history must be a non-empty [samples, time] array")
        if values.shape[1] <= self.seasonal_steps:
            fallback = values[:, -1:]
        else:
            fallback = values[:, -self.seasonal_steps : -self.seasonal_steps + horizon_steps]
            if fallback.shape[1] < horizon_steps:
                fallback = np.pad(
                    fallback,
                    ((0, 0), (0, horizon_steps - fallback.shape[1])),
                    mode="edge",
                )
        return fallback


def sklearn_model_suite(
    random_state: int = 42, *, forest_estimators: int = 100
) -> dict[str, DirectForecaster]:
    """Construct the reproducible baseline suite without fitting it."""

    return {
        "linear_regression": LinearRegressionForecaster(),
        "decision_tree": DecisionTreeForecaster(random_state=random_state),
        "random_forest": RandomForestForecaster(
            n_estimators=forest_estimators,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingForecaster(random_state=random_state),
        "lightgbm": LightGBMForecaster(random_state=random_state),
    }
