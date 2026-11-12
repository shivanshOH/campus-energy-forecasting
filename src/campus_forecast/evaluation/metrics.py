"""Central metric library for point and probabilistic forecasts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _paired_arrays(actual: object, predicted: object) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("actual and predicted must have equal length")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def mae(actual: object, predicted: object) -> float | None:
    y_true, y_pred = _paired_arrays(actual, predicted)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else None


def rmse(actual: object, predicted: object) -> float | None:
    y_true, y_pred = _paired_arrays(actual, predicted)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred)))) if len(y_true) else None


def cv_rmse(actual: object, predicted: object) -> float | None:
    y_true, y_pred = _paired_arrays(actual, predicted)
    if not len(y_true):
        return None
    mean_actual = float(np.mean(y_true))
    if np.isclose(mean_actual, 0):
        return None
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))) / abs(mean_actual) * 100)


def wape(actual: object, predicted: object) -> float | None:
    y_true, y_pred = _paired_arrays(actual, predicted)
    if not len(y_true):
        return None
    denominator = float(np.sum(np.abs(y_true)))
    if np.isclose(denominator, 0):
        return None
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


def r2(actual: object, predicted: object) -> float | None:
    y_true, y_pred = _paired_arrays(actual, predicted)
    if not len(y_true):
        return None
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    if np.isclose(denominator, 0):
        return 0.0
    return float(1 - np.sum(np.square(y_true - y_pred)) / denominator)


def evaluate_point(actual: object, predicted: object) -> dict[str, float | None]:
    """Return errors in original target units except percentage metrics."""

    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "cv_rmse_percent": cv_rmse(actual, predicted),
        "wape_percent": wape(actual, predicted),
        "r2": r2(actual, predicted),
    }


def pinball_loss(actual: object, predicted_quantile: object, quantile: float) -> float | None:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    y_true, y_pred = _paired_arrays(actual, predicted_quantile)
    if not len(y_true):
        return None
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_coverage(actual: object, lower: object, upper: object) -> float | None:
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_lower = np.asarray(lower, dtype=float).reshape(-1)
    y_upper = np.asarray(upper, dtype=float).reshape(-1)
    if not (len(y_true) == len(y_lower) == len(y_upper)):
        raise ValueError("interval inputs must have equal length")
    mask = np.isfinite(y_true) & np.isfinite(y_lower) & np.isfinite(y_upper)
    if not mask.any():
        return None
    return float(np.mean((y_true[mask] >= y_lower[mask]) & (y_true[mask] <= y_upper[mask])))


def mean_interval_width(lower: object, upper: object) -> float | None:
    y_lower, y_upper = _paired_arrays(lower, upper)
    if not len(y_lower):
        return None
    return float(np.mean(y_upper - y_lower))


def winkler_score(
    actual: object,
    lower: object,
    upper: object,
    *,
    coverage_level: float = 0.90,
) -> float | None:
    if not 0 < coverage_level < 1:
        raise ValueError("coverage_level must be between 0 and 1")
    alpha = 1 - coverage_level
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_lower = np.asarray(lower, dtype=float).reshape(-1)
    y_upper = np.asarray(upper, dtype=float).reshape(-1)
    if not (len(y_true) == len(y_lower) == len(y_upper)):
        raise ValueError("interval inputs must have equal length")
    mask = np.isfinite(y_true) & np.isfinite(y_lower) & np.isfinite(y_upper)
    if not mask.any():
        return None
    y_true, y_lower, y_upper = y_true[mask], y_lower[mask], y_upper[mask]
    score = y_upper - y_lower
    below = y_true < y_lower
    above = y_true > y_upper
    score = score + below * (2 / alpha) * (y_lower - y_true)
    score = score + above * (2 / alpha) * (y_true - y_upper)
    return float(np.mean(score))


def evaluate_probabilistic(
    actual: object,
    p10: object,
    p50: object,
    p90: object,
) -> dict[str, float | None]:
    return {
        "pinball_p10": pinball_loss(actual, p10, 0.10),
        "pinball_p50": pinball_loss(actual, p50, 0.50),
        "pinball_p90": pinball_loss(actual, p90, 0.90),
        "interval_coverage_80": interval_coverage(actual, p10, p90),
        "mean_interval_width": mean_interval_width(p10, p90),
        "winkler_80": winkler_score(actual, p10, p90, coverage_level=0.80),
    }


def evaluate_by_horizon(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, float | None]]:
    """Evaluate [samples, horizons] arrays separately at each horizon."""

    y_true = np.asarray(actual)
    y_pred = np.asarray(predicted)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("actual and predicted must be equal-shape two-dimensional arrays")
    return {
        f"horizon_{index + 1}": evaluate_point(y_true[:, index], y_pred[:, index])
        for index in range(y_true.shape[1])
    }


def block_bootstrap_ci(
    actual: object,
    predicted: object,
    statistic: Callable[[np.ndarray, np.ndarray], float | None] = lambda y, p: mae(y, p),
    *,
    blocks: object | None = None,
    repetitions: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Bootstrap complete blocks (e.g. days) instead of ten-minute rows."""

    y_true, y_pred = _paired_arrays(actual, predicted)
    if blocks is None:
        block_array = np.arange(len(y_true))
    else:
        block_array = np.asarray(blocks).reshape(-1)
        if len(block_array) != len(y_true):
            raise ValueError("blocks must have the same length as actual")
    unique_blocks = np.unique(block_array)
    if not len(unique_blocks):
        return {"estimate": None, "lower": None, "upper": None, "repetitions": 0}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        indices = np.concatenate([np.flatnonzero(block_array == block) for block in sampled_blocks])
        value = statistic(y_true[indices], y_pred[indices])
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return {
            "estimate": statistic(y_true, y_pred),
            "lower": None,
            "upper": None,
            "repetitions": 0,
        }
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(statistic(y_true, y_pred)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "repetitions": len(values),
    }


def day_blocks(timestamps: object) -> np.ndarray:
    values = pd.to_datetime(timestamps, utc=True)
    return values.strftime("%Y-%m-%d").to_numpy()
