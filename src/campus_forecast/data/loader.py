"""Safe, alias-aware loaders for I-BLEND-like tabular sources."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from campus_forecast.data.units import convert_energy_measurements
from campus_forecast.exceptions import DataValidationError
from campus_forecast.schema import EnergyMeasurementConfig

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".json", ".jsonl"}


def _normalise_column(name: object) -> str:
    return "_".join(str(name).strip().lower().replace("%", "pct").split())


ALIASES: dict[str, set[str]] = {
    "timestamp": {
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "date",
        "unix_timestamp",
        "unix_time",
        "epoch",
        "epoch_seconds",
    },
    "building_id": {"building_id", "building", "meter", "site", "location"},
    "energy_wh": {"energy_wh", "energy_(wh)", "energy_wh_", "consumption_wh"},
    "energy_kwh": {"energy_kwh", "energy_(kwh)", "consumption_kwh"},
    "power_w": {"power_w", "power_(watts)"},
    "power_kw": {"power_kw", "power_(kw)"},
    "occupancy": {
        "occupancy",
        "occupancy_count",
        "occupancy_(#)",
        "occupants",
        "count",
    },
    "temperature_c": {
        "temperature",
        "temperature_c",
        "temp",
        "temp_c",
        "temperature_(c)",
    },
    "humidity_pct": {
        "humidity",
        "humidity_pct",
        "relative_humidity",
        "relative_humidity_pct",
        "humidity_(%)",
    },
    "working_day": {
        "working_day",
        "working",
        "is_working",
        "calendar",
        "workday",
        "working_or_non_working",
    },
    "high_activity": {
        "high_activity",
        "activity",
        "semester",
        "high_activity_period",
        "high_or_low_activity",
    },
}


def _alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            mapping[_normalise_column(alias)] = canonical
    return mapping


_COLUMN_ALIAS_MAP = _alias_map()


@dataclass(frozen=True)
class DatasetPaths:
    """Optional explicit paths; ambiguous discovery is never silently resolved."""

    energy: tuple[Path, ...] = ()
    occupancy: tuple[Path, ...] = ()
    weather: tuple[Path, ...] = ()
    calendar: tuple[Path, ...] = ()


def discover_files(root: str | Path) -> dict[str, list[Path]]:
    """Discover files by conservative filename hints for an audit preview."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if root_path.is_file():
        candidates = [root_path]
    else:
        candidates = sorted(
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )

    result = {"energy": [], "occupancy": [], "weather": [], "calendar": [], "unknown": []}
    for path in candidates:
        # I-BLEND is commonly organized as energy/Academic.csv and
        # occupancy/Academic.csv, so directory names are part of the hint.
        name = path.stem.lower()
        parent_hint = path.parent.name.lower()
        is_occupancy_dir = parent_hint in {
            "occupancy",
            "wifi",
            "presence",
        } or parent_hint.startswith(("occupancy_", "wifi_", "presence_"))
        is_weather_dir = parent_hint in {
            "weather",
            "temperature",
            "humidity",
            "meteo",
        } or parent_hint.startswith(("weather_", "temperature_", "humidity_", "meteo_"))
        is_calendar_dir = parent_hint in {
            "calendar",
            "academic",
            "holiday",
        } or parent_hint.startswith(("calendar_", "academic_", "holiday_"))
        is_energy_dir = parent_hint in {
            "energy",
            "power",
            "meter",
            "transformer",
        } or parent_hint.startswith(("energy_", "power_", "meter_", "transformer_"))
        if is_occupancy_dir:
            kind = "occupancy"
        elif is_weather_dir:
            kind = "weather"
        elif is_calendar_dir:
            kind = "calendar"
        elif is_energy_dir:
            kind = "energy"
        elif any(token in name for token in ("occup", "wifi", "presence")):
            kind = "occupancy"
        elif any(token in name for token in ("weather", "temperature", "humidity", "meteo")):
            kind = "weather"
        elif any(token in name for token in ("calendar", "holiday")):
            kind = "calendar"
        elif any(token in name for token in ("energy", "power", "meter", "transformer")):
            kind = "energy"
        else:
            kind = "unknown"
        result[kind].append(path)
    return result


def _check_file(path: Path, max_bytes: int | None = None) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Source file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DataValidationError(
            f"Unsupported source extension {path.suffix!r}; allowed: {sorted(SUPPORTED_SUFFIXES)}"
        )
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise DataValidationError(f"Source file exceeds the configured size limit: {path}")


def read_table(path: str | Path, max_bytes: int | None = None) -> pd.DataFrame:
    """Read a supported tabular file without evaluating user-provided code."""

    source = Path(path)
    _check_file(source, max_bytes=max_bytes)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".jsonl":
        return pd.read_json(source, lines=True)
    return pd.read_json(source)


def parse_timestamp(values: Iterable[object], timezone: str = "Asia/Kolkata") -> pd.Series:
    """Parse Unix or textual timestamps into a timezone-aware series.

    Numeric timestamps are interpreted as Unix seconds unless their magnitude
    clearly indicates milliseconds. Text timestamps without an offset are
    interpreted as UTC before conversion; this conservative behavior avoids
    accidentally using local machine timezone settings.
    """

    series = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        return parsed.dt.tz_convert(timezone)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and len(numeric):
        magnitude = float(numeric.abs().median())
        unit = "ms" if magnitude > 10**11 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_convert(timezone)


def canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map source aliases while preserving unrecognized columns."""

    renamed: dict[object, str] = {}
    seen: set[str] = set()
    for original in frame.columns:
        normalized = _normalise_column(original)
        canonical = _COLUMN_ALIAS_MAP.get(normalized, str(original).strip())
        if canonical in seen:
            # Retain both source columns rather than silently overwriting data.
            canonical = f"{canonical}__duplicate"
        renamed[original] = canonical
        seen.add(canonical)
    return frame.rename(columns=renamed)


def normalize_source(
    frame: pd.DataFrame,
    *,
    source_kind: str,
    building_id: str | None = None,
    timezone: str = "Asia/Kolkata",
    energy_config: EnergyMeasurementConfig | None = None,
) -> pd.DataFrame:
    """Convert one source table into canonical names and validate timestamps."""

    if frame.empty:
        raise DataValidationError("Source table is empty")
    original = frame.copy()
    output = canonicalise_columns(frame.copy())
    if "timestamp" not in output:
        raise DataValidationError(f"{source_kind} source must contain a timestamp column")
    parsed_timestamp = parse_timestamp(output["timestamp"], timezone=timezone)
    valid_timestamp = parsed_timestamp.notna().to_numpy()
    output["timestamp"] = parsed_timestamp
    output = output.loc[valid_timestamp].copy()
    if output.empty:
        raise DataValidationError(f"{source_kind} source contains no parseable timestamps")

    if building_id is not None:
        output["building_id"] = str(building_id)
    elif "building_id" in output:
        output["building_id"] = output["building_id"].astype(str)
    elif source_kind in {"energy", "occupancy"}:
        raise DataValidationError(
            f"{source_kind} source needs a building_id column or an explicit building_id argument"
        )

    if source_kind == "energy":
        if energy_config is None:
            raise DataValidationError(
                "EnergyMeasurementConfig is required; source units and measurement semantics "
                "must be explicit before loading research energy data"
            )
        # Keep the original row alignment while discarding unparseable
        # timestamps. Conversion must never pair a measurement with a
        # timestamp from a different source row.
        source_frame = original.loc[valid_timestamp].copy().reset_index(drop=True)
        source_frame["timestamp"] = output["timestamp"].to_numpy()
        source_frame["building_id"] = output["building_id"].to_numpy()
        source_frame = source_frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
        output = output.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
        converted, _ = convert_energy_measurements(source_frame, energy_config)
        # Preserve canonical non-energy columns and all source provenance.
        for column in converted.columns:
            if (
                column not in output.columns
                or column.startswith("_")
                or column
                in {
                    "source_value",
                    "source_measurement_unit",
                    "source_measurement_semantics",
                    "source_measurement_column",
                    "energy_target_unit",
                    "energy_conversion_metadata",
                    energy_config.target_column,
                }
            ):
                output[column] = converted[column].to_numpy()
        output.attrs.update(converted.attrs)
    if source_kind == "occupancy" and "occupancy" not in output:
        raise DataValidationError("Occupancy source needs an occupancy column")
    if source_kind == "weather" and not {
        "temperature_c",
        "humidity_pct",
    }.intersection(output.columns):
        raise DataValidationError("Weather source needs temperature or humidity")

    output = output.sort_values(
        ["building_id", "timestamp"] if "building_id" in output else "timestamp"
    )
    return output.reset_index(drop=True)


class IBlendLoader:
    """Loader for explicit I-BLEND source files.

    The loader intentionally does not infer a unit conversion. The published
    data contains power-related columns while DECODE describes its target as
    energy/Wh; the source-unit metadata must therefore be supplied by the
    experiment configuration.
    """

    def __init__(
        self,
        *,
        timezone: str = "Asia/Kolkata",
        max_file_bytes: int | None = None,
    ) -> None:
        self.timezone = timezone
        self.max_file_bytes = max_file_bytes

    def load(
        self,
        path: str | Path,
        *,
        source_kind: str,
        building_id: str | None = None,
        energy_config: EnergyMeasurementConfig | None = None,
    ) -> pd.DataFrame:
        frame = read_table(path, max_bytes=self.max_file_bytes)
        return normalize_source(
            frame,
            source_kind=source_kind,
            building_id=building_id,
            timezone=self.timezone,
            energy_config=energy_config,
        )

    def load_many(
        self,
        paths: Iterable[str | Path],
        *,
        source_kind: str,
        building_ids: Iterable[str | None] | None = None,
    ) -> pd.DataFrame:
        path_list = list(paths)
        ids = list(building_ids) if building_ids is not None else [None] * len(path_list)
        if len(ids) != len(path_list):
            raise ValueError("building_ids must have the same length as paths")
        frames = [
            self.load(path, source_kind=source_kind, building_id=building_id)
            for path, building_id in zip(path_list, ids, strict=True)
        ]
        if not frames:
            raise DataValidationError(f"No {source_kind} paths were supplied")
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def _resample_by_building(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    interval: str,
    aggregation: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows: list[pd.DataFrame] = []
    for building_id, group in frame.groupby("building_id", sort=False):
        indexed = group.set_index("timestamp").sort_index()
        available = [column for column in value_columns if column in indexed]
        if not available:
            continue
        if aggregation == "max":
            values = indexed[available].resample(interval).max()
        elif aggregation == "sum":
            values = indexed[available].resample(interval).sum(min_count=1)
        else:
            values = indexed[available].resample(interval).mean()
        values["building_id"] = str(building_id)
        rows.append(values.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def merge_modalities(
    energy: pd.DataFrame,
    occupancy: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
    *,
    interval: str = "10min",
    energy_aggregation: str = "mean",
    energy_config: EnergyMeasurementConfig | None = None,
) -> pd.DataFrame:
    """Align sources on a common grid without filling unavailable observations."""

    if energy_config is None:
        # DataFrames created directly by tests may already contain explicitly
        # converted energy_wh/energy_kwh values. Source-file workflows must
        # provide a config through IBlendLoader before reaching this function.
        energy_column = "energy_wh" if "energy_wh" in energy.columns else "energy_kwh"
        if energy_column not in energy:
            raise DataValidationError(
                "merge_modalities requires explicit energy_config or converted energy column"
            )
        converted_energy = energy.copy()
        converted_energy.attrs.setdefault(
            "energy_target_unit", "Wh" if energy_column == "energy_wh" else "kWh"
        )
    else:
        if {
            energy_config.target_column,
            "source_value",
            "source_measurement_unit",
        }.issubset(energy.columns):
            converted_energy = energy.copy()
            energy_column = energy_config.target_column
        else:
            converted_energy, summary = convert_energy_measurements(energy, energy_config)
            converted_energy.attrs["energy_conversion_summary"] = summary.to_dict()
            energy_column = energy_config.target_column
    energy_grid = _resample_by_building(
        converted_energy,
        value_columns=[energy_column],
        interval=interval,
        aggregation="sum",
    )
    if energy_grid.empty:
        raise DataValidationError("No energy observations could be aligned")
    occupancy_grid = (
        _resample_by_building(
            occupancy,
            value_columns=["occupancy"],
            interval=interval,
            aggregation="max",
        )
        if occupancy is not None and not occupancy.empty
        else pd.DataFrame()
    )
    weather_grid = (
        _resample_by_building(
            weather.assign(building_id=weather.get("building_id", "campus")),
            value_columns=["temperature_c", "humidity_pct"],
            interval=interval,
            aggregation="mean",
        )
        if weather is not None and not weather.empty
        else pd.DataFrame()
    )

    buildings = sorted(energy_grid["building_id"].astype(str).unique())
    merged: list[pd.DataFrame] = []
    for building_id in buildings:
        base = energy_grid.loc[energy_grid["building_id"].astype(str) == building_id].copy()
        base = base.set_index("timestamp").sort_index()
        for source in (occupancy_grid, weather_grid):
            if not source.empty:
                source_building = source.loc[
                    source["building_id"].astype(str).isin([building_id, "campus"])
                ].copy()
                if not source_building.empty:
                    source_building = source_building.drop(columns=["building_id"]).set_index(
                        "timestamp"
                    )
                    base = base.join(source_building, how="left", rsuffix="_source")

        if calendar is not None and not calendar.empty:
            calendar_copy = calendar.copy()
            calendar_copy["date"] = calendar_copy["timestamp"].dt.date
            calendar_copy = calendar_copy.drop_duplicates("date").set_index("date")
            dates = pd.Series(base.index.date, index=base.index, name="date")
            for column in ("working_day", "high_activity"):
                if column in calendar_copy:
                    base[column] = dates.map(calendar_copy[column])

        base["building_id"] = building_id
        merged.append(base.reset_index())

    result = pd.concat(merged, ignore_index=True).sort_values(["building_id", "timestamp"])
    result = result.reset_index(drop=True)
    result.attrs["energy_column"] = energy_column
    result.attrs["energy_target_unit"] = (
        energy_config.target_unit.value
        if energy_config is not None
        else ("Wh" if energy_column == "energy_wh" else "kWh")
    )
    result.attrs["energy_conversion_summary"] = (
        converted_energy.attrs.get("energy_conversion_summary", {})
        if energy_config is not None
        else energy.attrs.get("energy_conversion_summary", {})
    )
    result.attrs["energy_measurement_config"] = (
        energy_config.to_dict()
        if energy_config is not None
        else energy.attrs.get("energy_measurement_config", {})
    )
    return result
