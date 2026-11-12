"""Small synthetic software fixture, always labelled DEMO DATA."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from campus_forecast.evaluation.metrics import evaluate_point
from campus_forecast.models.baselines import LinearRegressionForecaster
from campus_forecast.pipeline.features import build_calendar_features, make_issue_table
from campus_forecast.pipeline.splits import chronological_split
from campus_forecast.schema import FeatureConfiguration
from campus_forecast.services import attach_seasonal_predicted_occupancy


def make_demo_frame(*, days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=days * 144, freq="10min", tz="UTC")
    rows: list[dict[str, object]] = []
    for building_index, building in enumerate(("DEMO_ACADEMIC", "DEMO_LIBRARY")):
        for timestamp in timestamps:
            hour = timestamp.hour + timestamp.minute / 60
            working = float(timestamp.dayofweek < 5)
            occupancy = max(
                0,
                25 + 40 * working * max(0, np.sin(np.pi * (hour - 7) / 12)) + rng.normal(0, 3),
            )
            temperature = 24 + 6 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.4)
            energy = (
                180 + building_index * 50 + 2.8 * occupancy + 8 * temperature + rng.normal(0, 8)
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "building_id": building,
                    "energy_wh": max(0.0, energy),
                    "occupancy": occupancy,
                    "temperature_c": temperature,
                    "humidity_pct": 55 + rng.normal(0, 4),
                    "working_day": working,
                    "high_activity": working,
                }
            )
    return pd.DataFrame(rows)


def run_demo(*, seed: int = 42) -> dict[str, Any]:
    frame = make_demo_frame(seed=seed)
    frame = attach_seasonal_predicted_occupancy(frame)
    frame = build_calendar_features(frame)
    table = make_issue_table(
        frame,
        configuration=FeatureConfiguration.PREDICTED_OCCUPANCY,
        lookback_steps=24,
        horizon_steps=6,
    )
    _, _, _, boundaries = chronological_split(frame[["timestamp", "energy_wh"]].drop_duplicates())
    target_end = pd.to_datetime(table.target_timestamps.iloc[:, -1], utc=True)
    train = target_end <= boundaries.train_end
    test = target_end > boundaries.validation_end
    model = LinearRegressionForecaster()
    started = time.perf_counter()
    model.fit(table.features.loc[train].to_numpy(), table.targets.loc[train].to_numpy())
    prediction = model.predict(table.features.loc[test].to_numpy())
    actual = table.targets.loc[test].to_numpy()
    metrics = {
        f"horizon_{index + 1}_{name}": value
        for index in range(actual.shape[1])
        for name, value in evaluate_point(actual[:, index], prediction[:, index]).items()
    }
    return {
        "data_is_demo": True,
        "label": "DEMO DATA",
        "seed": seed,
        "rows": len(frame),
        "model": "linear_regression",
        "metrics": metrics,
        "duration_seconds": time.perf_counter() - started,
        "note": "Synthetic software fixture only; excluded from research-performance reports.",
    }
