"""Explicit physical-unit conversion for source energy measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from campus_forecast.exceptions import DataValidationError
from campus_forecast.schema import (
    EnergyMeasurementConfig,
    IrregularIntervalPolicy,
    MeasurementSemantics,
    MeasurementUnit,
)

POWER_TO_WATTS = {
    MeasurementUnit.W: 1.0,
    MeasurementUnit.KW: 1000.0,
}
ENERGY_TO_WH = {
    MeasurementUnit.WH: 1.0,
    MeasurementUnit.KWH: 1000.0,
}


@dataclass(frozen=True)
class ConversionSummary:
    source_column: str
    source_unit: str
    semantics: str
    target_unit: str
    aggregation: str
    intervals_used: int
    intervals_rejected: int
    last_interval_dropped: bool
    irregular_interval_policy: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _normalise_name(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("%", "pct").split())


def resolve_energy_column(frame: pd.DataFrame, config: EnergyMeasurementConfig) -> str:
    """Resolve an explicit source column or one unambiguous labelled candidate."""

    if config.source_column is not None:
        if config.source_column in frame.columns:
            return config.source_column
        wanted = _normalise_name(config.source_column)
        matches = [column for column in frame.columns if _normalise_name(column) == wanted]
        if len(matches) == 1:
            return str(matches[0])
        raise DataValidationError(
            f"Configured energy source column was not found: {config.source_column}"
        )

    candidates: list[str] = []
    for column in frame.columns:
        name = _normalise_name(column)
        if name in {
            "power",
            "power_w",
            "power_kw",
            "power_(watts)",
            "energy",
            "energy_wh",
            "energy_kwh",
            "energy_(wh)",
            "consumption",
            "consumption_wh",
            "consumption_kwh",
            "active_power",
        }:
            candidates.append(str(column))
    if len(candidates) != 1:
        raise DataValidationError(
            "Energy source_column must be explicit when the source has zero or multiple "
            f"energy candidates; candidates={candidates}"
        )
    return candidates[0]


def convert_energy_measurements(
    frame: pd.DataFrame,
    config: EnergyMeasurementConfig,
    *,
    timestamp_column: str = "timestamp",
    building_column: str = "building_id",
) -> tuple[pd.DataFrame, ConversionSummary]:
    """Convert source measurements into the requested energy target unit.

    For instantaneous power, each reading is integrated over the duration to
    the next observation. Irregular durations are handled according to the
    explicit policy in ``config``. The final reading is dropped unless an
    expected duration is explicitly requested.
    """

    required = {timestamp_column, building_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Energy conversion frame is missing columns: {missing}")
    source_column = (
        resolve_energy_column(frame, config) if "source_value" not in frame else "source_value"
    )
    output = frame.copy()
    output[timestamp_column] = pd.to_datetime(output[timestamp_column], utc=True)
    output["source_value"] = pd.to_numeric(output[source_column], errors="coerce")
    output["source_measurement_unit"] = config.source_unit.value
    output["source_measurement_semantics"] = config.semantics.value
    output["source_measurement_column"] = source_column
    output = output.sort_values([building_column, timestamp_column]).reset_index(drop=True)
    output["_conversion_status"] = "converted"
    output["_interval_minutes"] = np.nan
    output["_energy_wh_internal"] = np.nan
    rejected = 0
    used = 0
    last_dropped = False

    if config.semantics == MeasurementSemantics.INTERVAL_ENERGY:
        output["_energy_wh_internal"] = output["source_value"] * ENERGY_TO_WH[config.source_unit]
        used = int(output["_energy_wh_internal"].notna().sum())
    elif config.semantics == MeasurementSemantics.CUMULATIVE_ENERGY:
        factors = ENERGY_TO_WH[config.source_unit]
        for _, group in output.groupby(building_column, sort=False):
            indices = group.index
            differences = group["source_value"].diff() * factors
            negative = differences < 0
            output.loc[indices, "_energy_wh_internal"] = differences.to_numpy()
            output.loc[indices[negative.fillna(False)], "_conversion_status"] = (
                "counter_reset_or_decrease"
            )
            output.loc[indices[negative.fillna(False)], "_energy_wh_internal"] = np.nan
        used = int(output["_energy_wh_internal"].notna().sum())
    else:
        factor = POWER_TO_WATTS[config.source_unit]
        for _, group in output.groupby(building_column, sort=False):
            indices = group.index.to_numpy()
            times = group[timestamp_column]
            intervals = times.shift(-1).sub(times).dt.total_seconds().div(60)
            if (
                config.last_interval_policy == "expected_duration"
                and config.expected_interval_minutes
            ):
                intervals.iloc[-1] = config.expected_interval_minutes
            else:
                last_dropped = True
            for position, index in enumerate(indices):
                interval = intervals.iloc[position]
                output.loc[index, "_interval_minutes"] = interval
                if pd.isna(interval):
                    output.loc[index, "_conversion_status"] = "last_interval_dropped"
                    continue
                if config.expected_interval_minutes is not None:
                    is_gap = interval > config.expected_interval_minutes * config.gap_factor
                    if (
                        is_gap
                        and config.irregular_interval_policy == IrregularIntervalPolicy.REJECT_GAPS
                    ):
                        output.loc[index, "_conversion_status"] = "irregular_interval_rejected"
                        rejected += 1
                        continue
                    if (
                        is_gap
                        and config.irregular_interval_policy
                        == IrregularIntervalPolicy.CLIP_TO_EXPECTED
                    ):
                        interval = config.expected_interval_minutes
                        output.loc[index, "_conversion_status"] = "irregular_interval_clipped"
                if interval <= 0:
                    output.loc[index, "_conversion_status"] = "non_positive_interval_rejected"
                    rejected += 1
                    continue
                value = output.loc[index, "source_value"]
                if pd.isna(value):
                    output.loc[index, "_conversion_status"] = "missing_source_value"
                    continue
                output.loc[index, "_energy_wh_internal"] = value * factor * interval / 60.0
                used += 1

    target_factor = 1.0 if config.target_unit == MeasurementUnit.WH else 1 / 1000.0
    output[config.target_column] = output["_energy_wh_internal"] * target_factor
    output["energy_target_unit"] = config.target_unit.value
    output["energy_conversion_metadata"] = str(config.to_dict())
    summary = ConversionSummary(
        source_column=source_column,
        source_unit=config.source_unit.value,
        semantics=config.semantics.value,
        target_unit=config.target_unit.value,
        aggregation="sum_of_interval_energy",
        intervals_used=used,
        intervals_rejected=rejected,
        last_interval_dropped=last_dropped,
        irregular_interval_policy=config.irregular_interval_policy.value,
    )
    output.attrs["energy_measurement_config"] = config.to_dict()
    output.attrs["energy_conversion_summary"] = summary.to_dict()
    return output, summary
