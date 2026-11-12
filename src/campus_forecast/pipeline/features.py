"""Calendar, historical and future-known feature construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from campus_forecast.exceptions import DataValidationError, LeakageError
from campus_forecast.schema import FeatureConfiguration, OccupancyMode

CALENDAR_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "working_day",
    "high_activity",
]


def build_calendar_features(
    frame: pd.DataFrame, *, timestamp_column: str = "timestamp"
) -> pd.DataFrame:
    """Add cyclical calendar features known at forecast issue time and in advance."""

    if timestamp_column not in frame:
        raise DataValidationError(f"Missing timestamp column: {timestamp_column}")
    output = frame.copy()
    timestamps = pd.to_datetime(output[timestamp_column], utc=True)
    hour = timestamps.dt.hour + timestamps.dt.minute / 60
    dow = timestamps.dt.dayofweek
    month = timestamps.dt.month
    output["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    output["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    output["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    output["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    output["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    output["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    if "working_day" not in output:
        # A weekday fallback is reproducible but should be replaced by the
        # institute calendar for research runs.
        output["working_day"] = (dow < 5).astype(float)
    else:
        output["working_day"] = pd.to_numeric(output["working_day"], errors="coerce")
    if "high_activity" not in output:
        output["high_activity"] = output["working_day"]
    else:
        output["high_activity"] = pd.to_numeric(output["high_activity"], errors="coerce")
    return output


def feature_columns_for(
    configuration: FeatureConfiguration,
    *,
    energy_column: str = "energy_wh",
) -> list[str]:
    """Return sequence features for a configuration.

    Future occupancy is intentionally not included in this historical feature
    list; it is represented as a separately named future feature by
    :func:`make_issue_table` after an occupancy forecast is supplied.
    """

    if configuration == FeatureConfiguration.ENERGY_ONLY:
        return [energy_column]
    if configuration == FeatureConfiguration.ENERGY_CALENDAR:
        return [energy_column, *CALENDAR_COLUMNS]
    if configuration == FeatureConfiguration.ENERGY_WEATHER_CALENDAR:
        return [energy_column, "temperature_c", "humidity_pct", *CALENDAR_COLUMNS]
    if configuration == FeatureConfiguration.HISTORICAL_OCCUPANCY:
        return [energy_column, "occupancy", "temperature_c", "humidity_pct", *CALENDAR_COLUMNS]
    if configuration in {
        FeatureConfiguration.PREDICTED_OCCUPANCY,
        FeatureConfiguration.ORACLE_OCCUPANCY,
    }:
        return [energy_column, "temperature_c", "humidity_pct", *CALENDAR_COLUMNS]
    raise ValueError(f"Unsupported feature configuration: {configuration}")


@dataclass
class SupervisedIssueTable:
    features: pd.DataFrame
    targets: pd.DataFrame
    issue_timestamps: pd.Series
    target_timestamps: pd.DataFrame
    building_ids: pd.Series


def make_issue_table(
    frame: pd.DataFrame,
    *,
    configuration: FeatureConfiguration,
    lookback_steps: int = 144,
    horizon_steps: int = 1,
    interval_minutes: int = 10,
    occupancy_mode: OccupancyMode | None = None,
    occupancy_prediction_column: str = "occupancy_pred",
    predicted_occupancy: pd.DataFrame | None = None,
    energy_column: str = "energy_wh",
) -> SupervisedIssueTable:
    """Create direct multi-horizon tabular samples from issue-time information.

    The resulting rows contain historical lags, known future calendar fields,
    and, only for the explicit predicted/oracle configurations, future
    occupancy values. Future observed weather is never copied into features.
    """

    if configuration == FeatureConfiguration.PREDICTED_OCCUPANCY:
        occupancy_mode = occupancy_mode or OccupancyMode.PREDICTED
    elif configuration == FeatureConfiguration.ORACLE_OCCUPANCY:
        occupancy_mode = occupancy_mode or OccupancyMode.ORACLE
    else:
        occupancy_mode = occupancy_mode or OccupancyMode.NONE
    if (
        occupancy_mode == OccupancyMode.ORACLE
        and configuration != FeatureConfiguration.ORACLE_OCCUPANCY
    ):
        raise LeakageError("Oracle occupancy requires the explicit oracle feature configuration")
    if (
        occupancy_mode == OccupancyMode.PREDICTED
        and predicted_occupancy is None
        and occupancy_prediction_column not in frame
    ):
        raise DataValidationError(
            f"Predicted occupancy configuration requires {occupancy_prediction_column!r}"
        )
    required = {
        "timestamp",
        "building_id",
        energy_column,
        *feature_columns_for(configuration, energy_column=energy_column),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Issue table is missing columns: {missing}")

    prepared = build_calendar_features(frame).sort_values(["building_id", "timestamp"])
    features: list[dict[str, float | str]] = []
    targets: list[dict[str, float]] = []
    issues: list[pd.Timestamp] = []
    target_times: list[dict[str, pd.Timestamp]] = []
    buildings: list[str] = []
    expected_delta = pd.Timedelta(minutes=interval_minutes)
    historical_columns = feature_columns_for(configuration, energy_column=energy_column)
    future_calendar = [column for column in CALENDAR_COLUMNS]

    for building_id, group in prepared.groupby("building_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        timestamps = pd.to_datetime(group["timestamp"], utc=True)
        for end in range(lookback_steps - 1, len(group) - horizon_steps):
            history_times = timestamps.iloc[end - lookback_steps + 1 : end + 1]
            future_times = timestamps.iloc[end + 1 : end + horizon_steps + 1]
            all_times = pd.concat([history_times, future_times])
            if all_times.diff().dropna().ne(expected_delta).any():
                continue
            history = group.iloc[end - lookback_steps + 1 : end + 1]
            future = group.iloc[end + 1 : end + horizon_steps + 1]
            if history[historical_columns].isna().any().any() or future[energy_column].isna().any():
                continue
            # Calendar fields are known in advance but still need to be
            # present in the supplied calendar source.
            if future[future_calendar].isna().any().any():
                continue
            if occupancy_mode == OccupancyMode.ORACLE and future["occupancy"].isna().any():
                continue
            predicted_values: list[float] = []
            if occupancy_mode == OccupancyMode.PREDICTED:
                if predicted_occupancy is not None:
                    predicted_values = []
                    for step, target_timestamp in enumerate(future_times, start=1):
                        matches = predicted_occupancy.loc[
                            (predicted_occupancy["building_id"].astype(str) == str(building_id))
                            & (
                                pd.to_datetime(predicted_occupancy["issue_timestamp"], utc=True)
                                == timestamps.iloc[end]
                            )
                            & (
                                pd.to_datetime(predicted_occupancy["timestamp"], utc=True)
                                == target_timestamp
                            )
                            & (predicted_occupancy["horizon_step"] == step)
                        ]
                        if matches.empty:
                            predicted_values = []
                            break
                        predicted_values.append(float(matches.iloc[0]["occupancy_pred"]))
                    if len(predicted_values) != horizon_steps:
                        continue
                else:
                    if future[occupancy_prediction_column].isna().any():
                        continue
                    predicted_values = [
                        float(value) for value in future[occupancy_prediction_column]
                    ]

            row: dict[str, float | str] = {}
            for lag, (_, historic_row) in enumerate(history.iloc[::-1].iterrows()):
                for column in historical_columns:
                    row[f"{column}_lag_{lag}"] = float(historic_row[column])
            for step, (_, future_row) in enumerate(future.iterrows(), start=1):
                for column in future_calendar:
                    row[f"future_{column}_h{step}"] = float(future_row[column])
                if occupancy_mode == OccupancyMode.PREDICTED:
                    row[f"future_occupancy_h{step}"] = predicted_values[step - 1]
                elif occupancy_mode == OccupancyMode.ORACLE:
                    row[f"future_occupancy_h{step}"] = float(future_row["occupancy"])
            features.append(row)
            targets.append(
                {
                    f"energy_h{step}": float(value)
                    for step, value in enumerate(future[energy_column], 1)
                }
            )
            issues.append(pd.Timestamp(timestamps.iloc[end]))
            target_times.append(
                {
                    f"energy_h{step}": pd.Timestamp(value)
                    for step, value in enumerate(future_times, 1)
                }
            )
            buildings.append(str(building_id))

    target_frame = pd.DataFrame(targets)
    feature_frame = pd.DataFrame(features)
    target_time_frame = pd.DataFrame(target_times)
    if not feature_frame.empty:
        feature_frame = feature_frame.astype(float)
    return SupervisedIssueTable(
        features=feature_frame,
        targets=target_frame,
        issue_timestamps=pd.Series(pd.to_datetime(issues, utc=True), name="issue_timestamp"),
        target_timestamps=target_time_frame,
        building_ids=pd.Series(buildings, name="building_id", dtype="string"),
    )
