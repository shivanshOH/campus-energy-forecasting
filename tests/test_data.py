from __future__ import annotations

import pandas as pd

from campus_forecast.data.audit import audit_frame
from campus_forecast.data.loader import merge_modalities, normalize_source
from campus_forecast.schema import EnergyMeasurementConfig, MeasurementSemantics, MeasurementUnit


def test_unix_timestamp_and_aliases_are_canonicalized() -> None:
    frame = pd.DataFrame(
        {
            "Unix timestamp": [1704067200, 1704067260],
            "Power (watts)": [10, 12],
            "Building": ["A", "A"],
        }
    )
    normalized = normalize_source(
        frame,
        source_kind="energy",
        energy_config=EnergyMeasurementConfig(
            source_unit=MeasurementUnit.W,
            semantics=MeasurementSemantics.INSTANTANEOUS_POWER,
            expected_interval_minutes=1,
        ),
    )
    assert "power_w" in normalized.columns
    assert "energy_wh" in normalized.columns
    assert str(normalized["timestamp"].dt.tz) == "Asia/Kolkata"


def test_audit_counts_duplicates_gaps_and_invalid_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:30"]
            ),
            "building_id": ["A", "A", "A"],
            "energy": [1, -2, 3],
        }
    )
    report = audit_frame(
        frame, source_name="test", source_kind="energy", expected_interval_minutes=10
    )
    assert report.duplicate_timestamps == 1
    assert report.invalid_value_counts["energy_wh"] == 1
    assert any(issue.code == "missing_intervals" for issue in report.issues)


def test_merge_modalities_preserves_weather_missingness() -> None:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="10min", tz="UTC")
    energy = pd.DataFrame({"timestamp": timestamps, "building_id": "A", "energy_wh": range(6)})
    occupancy = pd.DataFrame({"timestamp": timestamps, "building_id": "A", "occupancy": range(6)})
    weather = pd.DataFrame(
        {
            "timestamp": timestamps[:3],
            "temperature_c": [20, 21, 22],
            "humidity_pct": [50, 51, 52],
            "building_id": "campus",
        }
    )
    merged = merge_modalities(energy, occupancy, weather)
    assert len(merged) == 6
    assert merged["temperature_c"].notna().sum() == 3
    assert merged["temperature_c"].isna().sum() == 3
