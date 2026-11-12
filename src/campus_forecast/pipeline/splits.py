"""Chronological split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from campus_forecast.exceptions import DataValidationError, LeakageError
from campus_forecast.schema import SplitBoundaries


def chronological_split(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitBoundaries]:
    """Perform the DECODE-style 70:15:15 chronological split."""

    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and sum to < 1")
    if timestamp_column not in frame:
        raise DataValidationError(f"Missing timestamp column: {timestamp_column}")
    ordered = frame.sort_values(timestamp_column).reset_index(drop=True)
    if ordered.empty:
        raise DataValidationError("Cannot split an empty frame")
    n = len(ordered)
    train_end_index = max(0, int(n * train_ratio) - 1)
    validation_end_index = max(train_end_index + 1, int(n * (train_ratio + validation_ratio)) - 1)
    validation_end_index = min(validation_end_index, n - 2 if n >= 2 else n - 1)
    train = ordered.iloc[: train_end_index + 1].copy()
    validation = ordered.iloc[train_end_index + 1 : validation_end_index + 1].copy()
    test = ordered.iloc[validation_end_index + 1 :].copy()
    if train.empty or validation.empty or test.empty:
        raise DataValidationError("Chronological split needs enough rows for train/validation/test")
    boundaries = SplitBoundaries(
        train_end=pd.Timestamp(train[timestamp_column].iloc[-1]),
        validation_end=pd.Timestamp(validation[timestamp_column].iloc[-1]),
        test_end=pd.Timestamp(test[timestamp_column].iloc[-1]),
    )
    return train, validation, test, boundaries


def split_by_boundaries(
    frame: pd.DataFrame,
    boundaries: SplitBoundaries,
    *,
    timestamp_column: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    ordered = frame.sort_values(timestamp_column).copy()
    train = ordered.loc[ordered[timestamp_column] <= boundaries.train_end]
    validation = ordered.loc[
        (ordered[timestamp_column] > boundaries.train_end)
        & (ordered[timestamp_column] <= boundaries.validation_end)
    ]
    test = ordered.loc[
        (ordered[timestamp_column] > boundaries.validation_end)
        & (ordered[timestamp_column] <= boundaries.test_end)
    ]
    if not (train.empty or validation.empty or test.empty):
        if train[timestamp_column].max() >= validation[timestamp_column].min():
            raise LeakageError("Train and validation periods overlap")
        if validation[timestamp_column].max() >= test[timestamp_column].min():
            raise LeakageError("Validation and test periods overlap")
    return {"train": train, "validation": validation, "test": test}


@dataclass(frozen=True)
class BuildingFold:
    target_building: str
    training_buildings: tuple[str, ...]
    adaptation_start: pd.Timestamp | None = None
    adaptation_end: pd.Timestamp | None = None
    evaluation_start: pd.Timestamp | None = None


def leave_one_building_out(
    frame: pd.DataFrame, *, building_column: str = "building_id"
) -> list[BuildingFold]:
    if building_column not in frame:
        raise DataValidationError(f"Missing building column: {building_column}")
    buildings = sorted(frame[building_column].astype(str).unique())
    return [
        BuildingFold(
            target_building=target, training_buildings=tuple(b for b in buildings if b != target)
        )
        for target in buildings
    ]
