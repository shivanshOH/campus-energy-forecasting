from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campus_forecast.data.loader import normalize_source
from campus_forecast.data.units import convert_energy_measurements
from campus_forecast.exceptions import DataValidationError
from campus_forecast.schema import (
    EnergyMeasurementConfig,
    IrregularIntervalPolicy,
    MeasurementSemantics,
    MeasurementUnit,
)


def test_power_is_integrated_using_actual_irregular_intervals() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:03"]
            ),
            "building_id": "A",
            "Power (watts)": [60.0, 60.0, 60.0],
        }
    )
    config = EnergyMeasurementConfig(
        source_unit=MeasurementUnit.W,
        semantics=MeasurementSemantics.INSTANTANEOUS_POWER,
        expected_interval_minutes=1,
    )
    converted, summary = convert_energy_measurements(frame, config)
    assert np.isclose(converted.loc[0, "energy_wh"], 1.0)
    assert np.isclose(converted.loc[1, "energy_wh"], 2.0)
    assert pd.isna(converted.loc[2, "energy_wh"])
    assert summary.last_interval_dropped is True


def test_kw_to_kwh_conversion_uses_duration() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:30"]),
            "building_id": "A",
            "load_kw": [1.0, 1.0],
        }
    )
    config = EnergyMeasurementConfig(
        source_column="load_kw",
        source_unit=MeasurementUnit.KW,
        semantics=MeasurementSemantics.INSTANTANEOUS_POWER,
        target_unit=MeasurementUnit.KWH,
        expected_interval_minutes=30,
        last_interval_policy="expected_duration",
    )
    converted, _ = convert_energy_measurements(frame, config)
    assert np.isclose(converted.loc[0, "energy_kwh"], 0.5)
    assert np.isclose(converted.loc[1, "energy_kwh"], 0.5)


def test_interval_and_cumulative_energy_units_are_distinguished() -> None:
    interval_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10"]),
            "building_id": "A",
            "interval_kwh": [1.5, 2.0],
        }
    )
    interval_config = EnergyMeasurementConfig(
        source_column="interval_kwh",
        source_unit=MeasurementUnit.KWH,
        semantics=MeasurementSemantics.INTERVAL_ENERGY,
        target_unit=MeasurementUnit.WH,
    )
    interval_output, _ = convert_energy_measurements(interval_frame, interval_config)
    assert interval_output["energy_wh"].tolist() == [1500.0, 2000.0]

    cumulative_frame = interval_frame.rename(columns={"interval_kwh": "meter_kwh"}).copy()
    cumulative_frame["meter_kwh"] = [10.0, 12.0]
    cumulative_config = EnergyMeasurementConfig(
        source_column="meter_kwh",
        source_unit=MeasurementUnit.KWH,
        semantics=MeasurementSemantics.CUMULATIVE_ENERGY,
        target_unit=MeasurementUnit.WH,
    )
    cumulative_output, _ = convert_energy_measurements(cumulative_frame, cumulative_config)
    assert pd.isna(cumulative_output.loc[0, "energy_wh"])
    assert cumulative_output.loc[1, "energy_wh"] == 2000.0


def test_irregular_gap_can_be_rejected_explicitly() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 03:00"]),
            "building_id": "A",
            "power": [100.0, 100.0],
        }
    )
    config = EnergyMeasurementConfig(
        source_column="power",
        source_unit=MeasurementUnit.W,
        semantics=MeasurementSemantics.INSTANTANEOUS_POWER,
        expected_interval_minutes=10,
        irregular_interval_policy=IrregularIntervalPolicy.REJECT_GAPS,
    )
    converted, summary = convert_energy_measurements(frame, config)
    assert pd.isna(converted.loc[0, "energy_wh"])
    assert summary.intervals_rejected == 1


def test_energy_loader_requires_explicit_semantics() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [1704067200],
            "Power (watts)": [100],
            "Building": ["A"],
        }
    )
    with pytest.raises(DataValidationError, match="EnergyMeasurementConfig"):
        normalize_source(frame, source_kind="energy")
