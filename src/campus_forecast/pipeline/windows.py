"""Direct multi-horizon window construction with temporal assertions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from campus_forecast.exceptions import DataValidationError, LeakageError


@dataclass
class WindowedArrays:
    features: np.ndarray
    targets: np.ndarray
    issue_timestamps: pd.DatetimeIndex
    target_timestamps: np.ndarray
    building_ids: np.ndarray
    feature_columns: list[str]

    def flatten(self) -> np.ndarray:
        return self.features.reshape(self.features.shape[0], -1)


def create_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str = "energy_wh",
    lookback_steps: int = 144,
    horizon_steps: int = 1,
    interval_minutes: int = 10,
    timestamp_column: str = "timestamp",
    building_column: str = "building_id",
) -> WindowedArrays:
    """Create direct windows using observations at or before each issue time.

    A sample ending at row `i` has features from `i-lookback+1:i+1` and its
    target is the vector at `i+1:i+horizon+1`. Samples crossing a gap or a
    building boundary are discarded rather than bridged with fabricated rows.
    """

    if lookback_steps < 1 or horizon_steps < 1:
        raise ValueError("lookback_steps and horizon_steps must be positive")
    required = set(feature_columns) | {target_column, timestamp_column, building_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Window frame is missing columns: {missing}")
    ordered = frame.sort_values([building_column, timestamp_column]).reset_index(drop=True)
    feature_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    issues: list[pd.Timestamp] = []
    target_times: list[np.ndarray] = []
    building_ids: list[str] = []
    expected_delta = pd.Timedelta(minutes=interval_minutes)

    for building_id, group in ordered.groupby(building_column, sort=False):
        group = group.sort_values(timestamp_column).reset_index(drop=True)
        timestamps = pd.to_datetime(group[timestamp_column], utc=True)
        values = group[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        targets = pd.to_numeric(group[target_column], errors="coerce").to_numpy(dtype=float)
        for end in range(lookback_steps - 1, len(group) - horizon_steps):
            history_times = timestamps.iloc[end - lookback_steps + 1 : end + 1]
            future_times = timestamps.iloc[end + 1 : end + horizon_steps + 1]
            full_times = pd.concat([history_times, future_times])
            if full_times.diff().dropna().ne(expected_delta).any():
                continue
            history = values[end - lookback_steps + 1 : end + 1]
            target = targets[end + 1 : end + horizon_steps + 1]
            if np.isnan(history).any() or np.isnan(target).any():
                continue
            feature_values.append(history)
            target_values.append(target)
            issues.append(pd.Timestamp(timestamps.iloc[end]))
            target_times.append(future_times.to_numpy())
            building_ids.append(str(building_id))

    if not feature_values:
        return WindowedArrays(
            features=np.empty((0, lookback_steps, len(feature_columns)), dtype=float),
            targets=np.empty((0, horizon_steps), dtype=float),
            issue_timestamps=pd.DatetimeIndex([], tz="UTC"),
            target_timestamps=np.empty((0, horizon_steps), dtype="datetime64[ns]"),
            building_ids=np.empty((0,), dtype=str),
            feature_columns=feature_columns,
        )
    result = WindowedArrays(
        features=np.stack(feature_values),
        targets=np.stack(target_values),
        issue_timestamps=pd.DatetimeIndex(issues),
        target_timestamps=np.stack(target_times),
        building_ids=np.asarray(building_ids),
        feature_columns=feature_columns,
    )
    if (result.target_timestamps <= result.issue_timestamps.to_numpy()[:, None]).any():
        raise LeakageError("Window target contains an observation at or before its issue time")
    return result


def filter_windows_to_period(
    windows: WindowedArrays,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> WindowedArrays:
    """Filter by target-period boundaries, preventing train targets leaking forward."""

    end_times = (
        pd.to_datetime(windows.target_timestamps[:, -1], utc=True)
        if len(windows.targets)
        else pd.DatetimeIndex([])
    )
    mask = np.ones(len(windows.targets), dtype=bool)
    if start is not None:
        mask &= end_times >= pd.Timestamp(start)
    if end is not None:
        mask &= end_times <= pd.Timestamp(end)
    return WindowedArrays(
        features=windows.features[mask],
        targets=windows.targets[mask],
        issue_timestamps=windows.issue_timestamps[mask],
        target_timestamps=windows.target_timestamps[mask],
        building_ids=windows.building_ids[mask],
        feature_columns=windows.feature_columns,
    )
