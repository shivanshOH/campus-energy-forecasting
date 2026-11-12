"""Causal alignment, imputation and train-only scaling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from campus_forecast.exceptions import DataValidationError, LeakageError
from campus_forecast.schema import ImputationStrategy


def align_to_grid(
    frame: pd.DataFrame,
    *,
    interval: str = "10min",
    timestamp_column: str = "timestamp",
    group_column: str = "building_id",
) -> pd.DataFrame:
    """Reindex each building to a regular grid and preserve missing readings."""

    if timestamp_column not in frame:
        raise DataValidationError(f"Missing timestamp column: {timestamp_column}")
    if group_column not in frame:
        working = frame.copy()
        working[group_column] = "campus"
    else:
        working = frame.copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], utc=True)
    rows: list[pd.DataFrame] = []
    for building_id, group in working.groupby(group_column, sort=False):
        group = group.sort_values(timestamp_column)
        if group.empty:
            continue
        index = pd.date_range(
            group[timestamp_column].iloc[0],
            group[timestamp_column].iloc[-1],
            freq=interval,
            tz="UTC",
        )
        indexed = group.set_index(timestamp_column).drop(columns=[group_column], errors="ignore")
        # Duplicate timestamps are retained through an explicit mean for numeric columns.
        if indexed.index.duplicated().any():
            numeric = indexed.select_dtypes(include="number").groupby(level=0).mean()
            non_numeric = (
                indexed.drop(columns=numeric.columns, errors="ignore").groupby(level=0).first()
            )
            indexed = non_numeric.join(numeric)
        reindexed = indexed.reindex(index)
        reindexed[group_column] = str(building_id)
        rows.append(reindexed.reset_index(names=timestamp_column))
    if not rows:
        return working.iloc[0:0].copy()
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values([group_column, timestamp_column])
        .reset_index(drop=True)
    )


@dataclass
class CausalImputer:
    """Imputer whose operational modes never inspect future rows.

    `fit` stores only training-derived initial values. `transform` carries the
    last observed value forward in the supplied order, including the final
    observed value from the fitted training frame when a validation/test frame
    begins with a missing value.
    """

    strategy: ImputationStrategy = ImputationStrategy.CAUSAL
    columns: list[str] | None = None
    add_indicators: bool = True
    max_forward_gap: int | None = None
    _initial_values: dict[str, float] = field(default_factory=dict, init=False)
    _columns: list[str] = field(default_factory=list, init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame) -> CausalImputer:
        if self.strategy == ImputationStrategy.BACKWARD_FILL_OFFLINE:
            raise LeakageError(
                "Backward fill is offline-only and cannot be fit for operational use"
            )
        if self.strategy == ImputationStrategy.TIME_INTERPOLATION_OFFLINE:
            raise LeakageError(
                "Time interpolation is offline-only and cannot be fit for operational use"
            )
        self._columns = list(self.columns or frame.select_dtypes(include="number").columns)
        self._initial_values = {}
        for column in self._columns:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                self._initial_values[column] = float(values.iloc[-1])
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise LeakageError("CausalImputer must be fit on training data before transform")
        output = frame.copy()
        for column in self._columns:
            if column not in output:
                continue
            missing = output[column].isna()
            if self.add_indicators:
                output[f"{column}__missing"] = missing.astype("int8")
            values = pd.to_numeric(output[column], errors="coerce").tolist()
            last = self._initial_values.get(column, np.nan)
            missing_run = 0
            for index, value in enumerate(values):
                if pd.isna(value):
                    missing_run += 1
                    if self.max_forward_gap is None or missing_run <= self.max_forward_gap:
                        values[index] = last
                else:
                    last = float(value)
                    missing_run = 0
            output[column] = values
        return output

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


@dataclass
class TrainOnlyMinMaxScaler:
    """A small serializable min-max scaler fitted only on training rows."""

    columns: list[str] | None = None
    _minimum: dict[str, float] = field(default_factory=dict, init=False)
    _maximum: dict[str, float] = field(default_factory=dict, init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame) -> TrainOnlyMinMaxScaler:
        self._minimum = {}
        self._maximum = {}
        self._columns = list(self.columns or frame.select_dtypes(include="number").columns)
        for column in self._columns:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.dropna()
            if valid.empty:
                self._minimum[column] = 0.0
                self._maximum[column] = 1.0
            else:
                self._minimum[column] = float(valid.min())
                self._maximum[column] = float(valid.max())
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise LeakageError("TrainOnlyMinMaxScaler must be fit before transform")
        output = frame.copy()
        for column in self._columns:
            if column not in output:
                continue
            minimum = self._minimum[column]
            maximum = self._maximum[column]
            span = maximum - minimum
            values = pd.to_numeric(output[column], errors="coerce")
            output[column] = 0.0 if span == 0 else (values - minimum) / span
        return output

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise LeakageError("TrainOnlyMinMaxScaler must be fit before inverse_transform")
        output = frame.copy()
        for column in self._columns:
            if column not in output:
                continue
            minimum = self._minimum[column]
            maximum = self._maximum[column]
            span = maximum - minimum
            output[column] = minimum if span == 0 else output[column] * span + minimum
        return output

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"minimum": self._minimum.copy(), "maximum": self._maximum.copy()}
