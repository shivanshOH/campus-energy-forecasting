"""Adapters for optional external validation datasets.

These adapters intentionally expose only the modalities documented by each
source. BDG2 is energy-only here; UCI occupancy is a room-level supplemental
occupancy experiment and is never merged with I-BLEND campus scores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from campus_forecast.data.loader import IBlendLoader, read_table
from campus_forecast.exceptions import DataValidationError
from campus_forecast.schema import EnergyMeasurementConfig


def load_bdg2_energy(
    path: str | Path,
    *,
    building_id: str | None = None,
    energy_config: EnergyMeasurementConfig | None = None,
) -> pd.DataFrame:
    """Load a Building Data Genome 2 energy table as an energy-only frame."""

    normalized = IBlendLoader().load(
        path,
        source_kind="energy",
        building_id=building_id,
        energy_config=energy_config,
    )
    if "building_id" not in normalized:
        raise DataValidationError("BDG2 energy input needs an explicit building_id")
    normalized["dataset_source"] = "building_data_genome_2"
    return normalized


def load_uci_occupancy(path: str | Path) -> pd.DataFrame:
    """Load the UCI room-level occupancy label as a separate supplement."""

    frame = read_table(path)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    timestamp_column = next(
        (columns[name] for name in ("date", "timestamp", "datetime") if name in columns),
        None,
    )
    occupancy_column = columns.get("occupancy")
    if timestamp_column is None or occupancy_column is None:
        raise DataValidationError("UCI occupancy input needs date/timestamp and Occupancy columns")
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce"),
            "occupancy": pd.to_numeric(frame[occupancy_column], errors="coerce"),
            "building_id": "uci_room_level",
            "dataset_source": "uci_occupancy_detection",
        }
    ).dropna(subset=["timestamp", "occupancy"])
    if output.empty:
        raise DataValidationError("UCI occupancy input has no valid observations")
    return output.reset_index(drop=True)
