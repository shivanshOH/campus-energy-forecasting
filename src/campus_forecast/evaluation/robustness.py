"""Controlled missing-data, outage and measurement-noise experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from campus_forecast.exceptions import LeakageError
from campus_forecast.pipeline.preprocessing import CausalImputer
from campus_forecast.schema import ImputationStrategy


@dataclass(frozen=True)
class CorruptionConfig:
    seed: int = 42
    occupancy_missing_percent: float = 0.0
    weather_missing_percent: float = 0.0
    outage_hours: int = 0
    outage_column_groups: tuple[str, ...] = ("occupancy", "temperature_c", "humidity_pct")
    occupancy_noise_std: float = 0.0
    drop_random_rows_percent: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mask_random(values: pd.Series, percentage: float, rng: np.random.Generator) -> pd.Series:
    if not 0 <= percentage <= 100:
        raise ValueError("missingness percentage must be in [0, 100]")
    mask = rng.random(len(values)) < percentage / 100
    output = values.copy()
    output.loc[mask] = np.nan
    return output


def corrupt_frame(
    frame: pd.DataFrame,
    config: CorruptionConfig,
    *,
    interval_minutes: int = 10,
) -> pd.DataFrame:
    """Apply reproducible corruption while retaining a full audit configuration."""

    rng = np.random.default_rng(config.seed)
    output = frame.copy()
    for column in ("occupancy",):
        if column in output and config.occupancy_missing_percent:
            output[column] = _mask_random(output[column], config.occupancy_missing_percent, rng)
    for column in ("temperature_c", "humidity_pct"):
        if column in output and config.weather_missing_percent:
            output[column] = _mask_random(output[column], config.weather_missing_percent, rng)
    if config.occupancy_noise_std and "occupancy" in output:
        noise = rng.normal(0, config.occupancy_noise_std, size=len(output))
        output["occupancy"] = np.maximum(
            0.0, pd.to_numeric(output["occupancy"], errors="coerce") + noise
        )

    if config.outage_hours > 0:
        outage_steps = max(1, int(config.outage_hours * 60 / interval_minutes))
        group_column = "building_id" if "building_id" in output else None
        groups = output.groupby(group_column, sort=False) if group_column else [("campus", output)]
        for _, group in groups:
            if len(group) <= outage_steps:
                start = 0
            else:
                start = int(rng.integers(0, len(group) - outage_steps + 1))
            selected_index = group.iloc[start : start + outage_steps].index
            for column in config.outage_column_groups:
                if column in output:
                    output.loc[selected_index, column] = np.nan

    if config.drop_random_rows_percent:
        if not 0 <= config.drop_random_rows_percent <= 100:
            raise ValueError("drop_random_rows_percent must be in [0, 100]")
        keep = rng.random(len(output)) >= config.drop_random_rows_percent / 100
        output = output.loc[keep].copy()

    return output.sort_values(
        ["building_id", "timestamp"] if "building_id" in output else "timestamp"
    ).reset_index(drop=True)


def impute_frame(
    frame: pd.DataFrame,
    *,
    strategy: ImputationStrategy,
    columns: list[str] | None = None,
    offline: bool = False,
    training_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply a named imputation policy and enforce operational boundaries."""

    selected = columns or [
        column for column in ("occupancy", "temperature_c", "humidity_pct") if column in frame
    ]
    if (
        strategy
        in {
            ImputationStrategy.BACKWARD_FILL_OFFLINE,
            ImputationStrategy.TIME_INTERPOLATION_OFFLINE,
        }
        and not offline
    ):
        raise LeakageError(f"{strategy} may only be used for explicitly offline analysis")
    output = frame.copy()
    if strategy == ImputationStrategy.NONE:
        return output
    if strategy == ImputationStrategy.CAUSAL or strategy == ImputationStrategy.FORWARD_FILL:
        fit_frame = training_frame if training_frame is not None else frame
        imputer = CausalImputer(strategy=ImputationStrategy.CAUSAL, columns=selected)
        imputer.fit(fit_frame)
        return imputer.transform(output)
    if strategy == ImputationStrategy.BACKWARD_FILL_OFFLINE:
        output[selected] = output[selected].bfill()
        return output
    if strategy == ImputationStrategy.TIME_INTERPOLATION_OFFLINE:
        indexed = output.set_index(pd.to_datetime(output["timestamp"], utc=True))
        indexed[selected] = indexed[selected].interpolate(method="time").bfill().ffill()
        return indexed.reset_index(drop=True)
    raise ValueError(f"Unknown imputation strategy: {strategy}")


def degradation_percent(clean_error: float | None, corrupted_error: float | None) -> float | None:
    if clean_error is None or corrupted_error is None or np.isclose(clean_error, 0):
        return None
    return float((corrupted_error - clean_error) / abs(clean_error) * 100)
