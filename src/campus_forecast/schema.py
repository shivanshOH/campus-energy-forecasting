"""Canonical data and experiment schemas.

The schema is deliberately independent of a particular dataset vendor. Source
loaders map aliases into these stable names and retain provenance in metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd


class DataSourceKind(StrEnum):
    ENERGY = "energy"
    OCCUPANCY = "occupancy"
    WEATHER = "weather"
    CALENDAR = "calendar"
    MERGED = "merged"


class OccupancyMode(StrEnum):
    NONE = "none"
    PREDICTED = "predicted"
    ORACLE = "oracle"


class ImputationStrategy(StrEnum):
    NONE = "none"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL_OFFLINE = "backward_fill_offline"
    TIME_INTERPOLATION_OFFLINE = "time_interpolation_offline"
    CAUSAL = "causal"


class MeasurementUnit(StrEnum):
    W = "W"
    KW = "kW"
    WH = "Wh"
    KWH = "kWh"


class MeasurementSemantics(StrEnum):
    INSTANTANEOUS_POWER = "instantaneous_power"
    INTERVAL_ENERGY = "interval_energy"
    CUMULATIVE_ENERGY = "cumulative_energy"


class IrregularIntervalPolicy(StrEnum):
    USE_ACTUAL_INTERVAL = "use_actual_interval"
    REJECT_GAPS = "reject_gaps"
    CLIP_TO_EXPECTED = "clip_to_expected"


@dataclass(frozen=True)
class EnergyMeasurementConfig:
    """Explicit source measurement semantics for energy conversion.

    A power reading is integrated over the actual timestamp-to-next-timestamp
    duration. Interval/cumulative energy readings are converted by unit factor
    only. No unit is inferred from magnitude.
    """

    source_unit: MeasurementUnit
    semantics: MeasurementSemantics
    target_unit: MeasurementUnit = MeasurementUnit.WH
    source_column: str | None = None
    expected_interval_minutes: float | None = None
    irregular_interval_policy: IrregularIntervalPolicy = IrregularIntervalPolicy.USE_ACTUAL_INTERVAL
    gap_factor: float = 1.5
    last_interval_policy: str = "drop"

    def __post_init__(self) -> None:
        if self.source_unit in {MeasurementUnit.W, MeasurementUnit.KW} and self.semantics not in {
            MeasurementSemantics.INSTANTANEOUS_POWER,
        }:
            raise ValueError("W/kW sources must use instantaneous_power semantics")
        if self.source_unit in {MeasurementUnit.WH, MeasurementUnit.KWH} and self.semantics not in {
            MeasurementSemantics.INTERVAL_ENERGY,
            MeasurementSemantics.CUMULATIVE_ENERGY,
        }:
            raise ValueError(
                "Wh/kWh sources must use interval_energy or cumulative_energy semantics"
            )
        if self.target_unit not in {MeasurementUnit.WH, MeasurementUnit.KWH}:
            raise ValueError("target_unit must be Wh or kWh")
        if self.expected_interval_minutes is not None and self.expected_interval_minutes <= 0:
            raise ValueError("expected_interval_minutes must be positive")
        if self.gap_factor <= 0:
            raise ValueError("gap_factor must be positive")
        if self.last_interval_policy not in {"drop", "expected_duration"}:
            raise ValueError("last_interval_policy must be 'drop' or 'expected_duration'")

    @property
    def target_column(self) -> str:
        return "energy_wh" if self.target_unit == MeasurementUnit.WH else "energy_kwh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_column": self.source_column,
            "source_unit": self.source_unit.value,
            "semantics": self.semantics.value,
            "target_unit": self.target_unit.value,
            "expected_interval_minutes": self.expected_interval_minutes,
            "irregular_interval_policy": self.irregular_interval_policy.value,
            "gap_factor": self.gap_factor,
            "last_interval_policy": self.last_interval_policy,
        }


class FeatureConfiguration(StrEnum):
    ENERGY_ONLY = "A_energy_only"
    ENERGY_CALENDAR = "B_energy_calendar"
    ENERGY_WEATHER_CALENDAR = "C_energy_weather_calendar"
    HISTORICAL_OCCUPANCY = "D_historical_occupancy"
    PREDICTED_OCCUPANCY = "E_predicted_occupancy"
    ORACLE_OCCUPANCY = "F_oracle_occupancy_diagnostic"


@dataclass(frozen=True)
class HorizonSpec:
    """A direct forecast horizon on a regular sampling grid."""

    name: str
    steps: int
    minutes: int


HORIZONS: dict[str, HorizonSpec] = {
    "10m": HorizonSpec("10m", 1, 10),
    "1h": HorizonSpec("1h", 6, 60),
    "6h": HorizonSpec("6h", 36, 360),
    "24h": HorizonSpec("24h", 144, 1440),
}


@dataclass
class AuditIssue:
    code: str
    severity: str
    message: str
    count: int | None = None
    examples: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    source_name: str
    source_kind: str
    rows: int
    columns: list[str]
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    inferred_interval_minutes: float | None = None
    expected_interval_minutes: float | None = None
    duplicate_timestamps: int = 0
    missing_timestamp_rows: int = 0
    missing_percentages: dict[str, float] = field(default_factory=dict)
    invalid_value_counts: dict[str, int] = field(default_factory=dict)
    buildings: list[str] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        return {
            "train_end": self.train_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


@dataclass
class ExperimentRecord:
    experiment_id: str
    dataset_id: str
    dataset_version: str
    building_id: str | None
    train_period: dict[str, str]
    validation_period: dict[str, str]
    test_period: dict[str, str]
    model_name: str
    model_architecture: dict[str, Any]
    hyperparameters: dict[str, Any]
    feature_configuration: str
    input_window_steps: int
    forecast_horizon_steps: int
    random_seed: int
    corruption_configuration: dict[str, Any]
    target_unit: str = MeasurementUnit.WH.value
    measurement_metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    training_duration_seconds: float | None = None
    checkpoint_path: str | None = None
    status: str = "created"
    data_is_demo: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data
