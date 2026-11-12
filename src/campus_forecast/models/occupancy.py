"""Occupancy forecasting models and causal prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from campus_forecast.exceptions import DataValidationError, LeakageError
from campus_forecast.models.baselines import DirectSklearnForecaster


@dataclass
class SeasonalNaiveOccupancyForecaster:
    seasonal_steps: int = 144
    interval_minutes: int = 10
    _history: pd.DataFrame | None = None

    def fit(self, frame: pd.DataFrame) -> SeasonalNaiveOccupancyForecaster:
        required = {"timestamp", "building_id", "occupancy"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataValidationError(f"Occupancy frame is missing columns: {missing}")
        self._history = frame.sort_values(["building_id", "timestamp"])[
            ["timestamp", "building_id", "occupancy"]
        ].copy()
        self._history["timestamp"] = pd.to_datetime(self._history["timestamp"], utc=True)
        return self

    def predict(
        self,
        issue_rows: pd.DataFrame,
        *,
        horizon_steps: int,
        allow_future_context: bool = False,
    ) -> pd.DataFrame:
        if self._history is None:
            raise RuntimeError("Occupancy forecaster has not been fitted")
        if allow_future_context:
            raise LeakageError("Operational occupancy prediction cannot use future context")
        required = {"timestamp", "building_id"}
        missing = sorted(required.difference(issue_rows.columns))
        if missing:
            raise DataValidationError(f"Issue rows are missing columns: {missing}")
        history = self._history
        result: list[dict[str, object]] = []
        for _, row in issue_rows.iterrows():
            issue = pd.Timestamp(row["timestamp"])
            building = str(row["building_id"])
            building_history = history.loc[history["building_id"].astype(str) == building]
            building_history = building_history.loc[building_history["timestamp"] <= issue]
            for step in range(1, horizon_steps + 1):
                target_time = issue + pd.Timedelta(minutes=self.interval_minutes * step)
                seasonal_time = target_time - pd.Timedelta(
                    minutes=self.interval_minutes * self.seasonal_steps
                )
                match = building_history.loc[
                    building_history["timestamp"] == seasonal_time, "occupancy"
                ]
                if match.empty:
                    # The fallback is deliberately bounded by the issue time.
                    # Looking up to target_time would expose actual future
                    # occupancy when a seasonal reference is unavailable.
                    match = building_history.loc[
                        building_history["timestamp"] <= issue, "occupancy"
                    ]
                value = float(match.iloc[-1]) if not match.empty else 0.0
                result.append(
                    {
                        "issue_timestamp": issue,
                        "timestamp": target_time,
                        "building_id": building,
                        "horizon_step": step,
                        "occupancy_pred": max(0.0, value),
                    }
                )
        return pd.DataFrame(result)


class TreeOccupancyForecaster:
    """Direct tree occupancy model using lagged occupancy and known calendar."""

    def __init__(self, *, lookback_steps: int = 144, random_state: int = 42) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.lookback_steps = lookback_steps
        self.random_state = random_state
        self._model = DirectSklearnForecaster(
            estimator_factory=lambda: HistGradientBoostingRegressor(random_state=random_state)
        )
        self._feature_columns: list[str] = []

    @staticmethod
    def _calendar(row: pd.Series) -> list[float]:
        timestamp = pd.Timestamp(row["timestamp"])
        hour = timestamp.hour + timestamp.minute / 60
        dow = timestamp.dayofweek
        return [
            float(np.sin(2 * np.pi * hour / 24)),
            float(np.cos(2 * np.pi * hour / 24)),
            float(np.sin(2 * np.pi * dow / 7)),
            float(np.cos(2 * np.pi * dow / 7)),
            float(dow < 5),
        ]

    def _make_table(self, frame: pd.DataFrame, horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        required = {"timestamp", "building_id", "occupancy"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataValidationError(f"Occupancy frame is missing columns: {missing}")
        rows: list[list[float]] = []
        targets: list[list[float]] = []
        expected = pd.Timedelta(minutes=10)
        for _, group in frame.sort_values(["building_id", "timestamp"]).groupby(
            "building_id", sort=False
        ):
            group = group.reset_index(drop=True)
            timestamps = pd.to_datetime(group["timestamp"], utc=True)
            occupancy = pd.to_numeric(group["occupancy"], errors="coerce").to_numpy()
            for end in range(self.lookback_steps - 1, len(group) - horizon_steps):
                history_times = timestamps.iloc[end - self.lookback_steps + 1 : end + 1]
                future_times = timestamps.iloc[end + 1 : end + horizon_steps + 1]
                if pd.concat([history_times, future_times]).diff().dropna().ne(expected).any():
                    continue
                history = occupancy[end - self.lookback_steps + 1 : end + 1]
                future = occupancy[end + 1 : end + horizon_steps + 1]
                if np.isnan(history).any() or np.isnan(future).any():
                    continue
                rows.append([*history[::-1], *self._calendar(group.iloc[end])])
                targets.append(list(future))
        if not rows:
            raise DataValidationError("No complete occupancy windows were available")
        self._feature_columns = [
            *[f"occupancy_lag_{i}" for i in range(self.lookback_steps)],
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "working_day",
        ]
        return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)

    def fit(self, frame: pd.DataFrame, *, horizon_steps: int = 1) -> TreeOccupancyForecaster:
        x, y = self._make_table(frame, horizon_steps)
        self._model.fit(x, y)
        return self

    def predict_from_windows(self, features: np.ndarray) -> np.ndarray:
        predictions = self._model.predict(features)
        return np.maximum(predictions, 0.0)

    def _prediction_features(
        self,
        frame: pd.DataFrame,
        issue_rows: pd.DataFrame,
    ) -> np.ndarray:
        required = {"timestamp", "building_id"}
        missing = sorted(required.difference(issue_rows.columns))
        if missing:
            raise DataValidationError(f"Occupancy issue rows are missing columns: {missing}")
        expected = pd.Timedelta(minutes=10)
        rows: list[list[float]] = []
        history = frame.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
        for _, issue_row in issue_rows.iterrows():
            issue = pd.Timestamp(issue_row["timestamp"])
            building = str(issue_row["building_id"])
            group = history.loc[
                (history["building_id"].astype(str) == building) & (history["timestamp"] <= issue)
            ].sort_values("timestamp")
            if len(group) < self.lookback_steps:
                raise DataValidationError(
                    "Not enough issue-time occupancy history for tree prediction"
                )
            group = group.tail(self.lookback_steps).reset_index(drop=True)
            if group["timestamp"].diff().dropna().ne(expected).any():
                raise DataValidationError(
                    "Occupancy issue-time history contains an irregular interval"
                )
            values = pd.to_numeric(group["occupancy"], errors="coerce").to_numpy(dtype=float)
            if np.isnan(values).any():
                raise DataValidationError("Occupancy issue-time history contains missing values")
            rows.append([*values[::-1], *self._calendar(pd.Series({"timestamp": issue}))])
        return np.asarray(rows, dtype=float)

    def predict(
        self,
        frame: pd.DataFrame,
        issue_rows: pd.DataFrame,
        *,
        horizon_steps: int,
    ) -> pd.DataFrame:
        features = self._prediction_features(frame, issue_rows)
        predictions = self.predict_from_windows(features)
        result: list[dict[str, object]] = []
        for row_index, (_, issue_row) in enumerate(issue_rows.iterrows()):
            issue = pd.Timestamp(issue_row["timestamp"])
            building = str(issue_row["building_id"])
            for step in range(1, horizon_steps + 1):
                result.append(
                    {
                        "issue_timestamp": issue,
                        "timestamp": issue + pd.Timedelta(minutes=10 * step),
                        "building_id": building,
                        "horizon_step": step,
                        "occupancy_pred": float(predictions[row_index, step - 1]),
                    }
                )
        return pd.DataFrame(result)
