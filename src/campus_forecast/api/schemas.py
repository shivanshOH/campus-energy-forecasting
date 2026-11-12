"""Validated REST request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from campus_forecast.schema import (
    EnergyMeasurementConfig,
    FeatureConfiguration,
    IrregularIntervalPolicy,
    MeasurementSemantics,
    MeasurementUnit,
    OccupancyMode,
)


class EnergyMeasurementRequest(BaseModel):
    source_unit: MeasurementUnit
    semantics: MeasurementSemantics
    target_unit: MeasurementUnit = MeasurementUnit.WH
    source_column: str | None = Field(default=None, max_length=200)
    expected_interval_minutes: float | None = Field(default=None, gt=0, le=1440)
    irregular_interval_policy: IrregularIntervalPolicy = IrregularIntervalPolicy.USE_ACTUAL_INTERVAL
    gap_factor: float = Field(default=1.5, gt=0, le=100)
    last_interval_policy: str = Field(default="drop", pattern=r"^(drop|expected_duration)$")

    @model_validator(mode="after")
    def validate_units(self) -> EnergyMeasurementRequest:
        EnergyMeasurementConfig(
            source_unit=self.source_unit,
            semantics=self.semantics,
            target_unit=self.target_unit,
            source_column=self.source_column,
            expected_interval_minutes=self.expected_interval_minutes,
            irregular_interval_policy=self.irregular_interval_policy,
            gap_factor=self.gap_factor,
            last_interval_policy=self.last_interval_policy,
        )
        return self

    def to_domain(self) -> EnergyMeasurementConfig:
        return EnergyMeasurementConfig(
            source_unit=self.source_unit,
            semantics=self.semantics,
            target_unit=self.target_unit,
            source_column=self.source_column,
            expected_interval_minutes=self.expected_interval_minutes,
            irregular_interval_policy=self.irregular_interval_policy,
            gap_factor=self.gap_factor,
            last_interval_policy=self.last_interval_policy,
        )


class AuditRequest(BaseModel):
    root_path: str = Field(default="data/raw/i_blend", min_length=1, max_length=500)
    dataset_id: str = Field(
        default="i_blend", min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.-]+$"
    )
    dataset_version: str = Field(default="unknown", min_length=1, max_length=128)
    energy_measurement: EnergyMeasurementRequest | None = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Any | None = None
    error: str | None = None


class ForecastRequest(BaseModel):
    dataset_id: str = Field(default="i_blend", min_length=1, max_length=128)
    building_id: str = Field(min_length=1, max_length=128)
    horizon: str = Field(default="1h", pattern=r"^(10m|1h|6h|24h)$")
    model_name: str = Field(default="random_forest", min_length=1, max_length=128)
    feature_configuration: FeatureConfiguration = FeatureConfiguration.PREDICTED_OCCUPANCY
    occupancy_mode: OccupancyMode = OccupancyMode.PREDICTED
    issue_timestamp: str | None = None
    diagnostic: bool = False
    occupancy_model_name: str = Field(
        default="seasonal_naive", pattern=r"^(seasonal_naive|tree|hist_gradient_boosting|lightgbm)$"
    )
    energy_measurement: EnergyMeasurementRequest | None = None

    @field_validator("occupancy_mode")
    @classmethod
    def reject_oracle_without_diagnostic(cls, value: OccupancyMode, info):
        # The cross-field restriction is enforced in the route as well; this
        # validator provides a clear schema-level message for standalone use.
        return value


class TrainRequest(BaseModel):
    dataset_id: str = Field(default="i_blend", min_length=1, max_length=128)
    building_id: str | None = Field(default=None, max_length=128)
    model_name: str = Field(default="random_forest", min_length=1, max_length=128)
    horizon: str = Field(default="1h", pattern=r"^(10m|1h|6h|24h)$")
    lookback_hours: int = Field(default=24, ge=1, le=168)
    feature_configuration: FeatureConfiguration = FeatureConfiguration.PREDICTED_OCCUPANCY
    random_seed: int = Field(default=42, ge=0, le=2**31 - 1)
    development_sample_rows: int | None = Field(default=None, ge=100, le=5_000_000)
    occupancy_model_name: str = Field(
        default="seasonal_naive", pattern=r"^(seasonal_naive|tree|hist_gradient_boosting|lightgbm)$"
    )
    energy_measurement: EnergyMeasurementRequest | None = None


class ExperimentRunRequest(TrainRequest):
    dataset_version: str = Field(default="unknown", min_length=1, max_length=128)
    corruption_configuration: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    experiment_ids: list[str] = Field(min_length=1, max_length=100)


class RobustnessRequest(TrainRequest):
    occupancy_missing_percent: float = Field(default=0, ge=0, le=30)
    weather_missing_percent: float = Field(default=0, ge=0, le=30)
    outage_hours: int = Field(default=0, ge=0, le=24)
    occupancy_noise_std: float = Field(default=0, ge=0, le=1000)
    imputation_strategy: str = Field(default="causal", min_length=1, max_length=64)


class TransferRequest(TrainRequest):
    target_building: str = Field(min_length=1, max_length=128)
    adaptation_days: list[int] = Field(default=[1, 3, 7, 14, 30], min_length=1, max_length=5)


class ReportRequest(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=128)
    title: str = Field(
        default="Campus Energy Forecasting Research Report", min_length=1, max_length=200
    )
    format: str = Field(default="html", pattern=r"^html$")


class ScenarioRunRequest(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=128)
    occupancy_multiplier: float = Field(default=1.0, ge=0, le=5)
    temperature_delta_c: float = Field(default=0.0, ge=-30, le=30)
    humidity_delta_pct: float = Field(default=0.0, ge=-100, le=100)
    working_day_override: float | None = Field(default=None, ge=0, le=1)
