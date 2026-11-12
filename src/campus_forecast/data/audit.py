"""Data-quality and data-availability audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from campus_forecast.data.loader import (
    canonicalise_columns,
    discover_files,
    parse_timestamp,
    read_table,
)
from campus_forecast.data.units import resolve_energy_column
from campus_forecast.schema import AuditIssue, DataQualityReport, EnergyMeasurementConfig

EXPECTED_INTERVALS = {"energy": 1, "occupancy": 10, "weather": 30, "calendar": 1440}


def _infer_building_from_path(path: Path) -> str:
    name = path.stem.strip().lower().replace(" ", "_")
    for token in ("energy", "power", "occupancy", "weather", "calendar"):
        name = name.replace(token, "")
    return name.strip("_-. ") or path.stem


def _safe_iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def audit_frame(
    frame: pd.DataFrame,
    *,
    source_name: str,
    source_kind: str,
    expected_interval_minutes: float | None = None,
    timezone: str = "Asia/Kolkata",
    energy_config: EnergyMeasurementConfig | None = None,
    building_id: str | None = None,
) -> DataQualityReport:
    """Audit one already-loaded or raw source frame."""

    canonical = canonicalise_columns(frame.copy())
    if "timestamp" in canonical:
        canonical["timestamp"] = parse_timestamp(canonical["timestamp"], timezone=timezone)
    timestamp = canonical.get("timestamp", pd.Series(dtype="datetime64[ns, UTC]"))
    valid_timestamp = timestamp.dropna().sort_values()
    expected = expected_interval_minutes or (
        energy_config.expected_interval_minutes
        if source_kind == "energy" and energy_config is not None
        else EXPECTED_INTERVALS.get(source_kind)
    )

    # Count redundant rows, not the first member of each duplicate group.
    duplicate_count = (
        int(timestamp.duplicated(keep="first").sum()) if "timestamp" in canonical else 0
    )
    missing_rows = int(timestamp.isna().sum()) if "timestamp" in canonical else len(canonical)
    issues: list[AuditIssue] = []
    measurement_metadata: dict[str, object] = {}
    if source_kind == "energy":
        if energy_config is None:
            issues.append(
                AuditIssue(
                    code="energy_unit_semantics_unresolved",
                    severity="error",
                    message=(
                        "Energy source units and measurement semantics are unresolved. "
                        "Supply EnergyMeasurementConfig before any research experiment."
                    ),
                )
            )
            measurement_metadata["research_ready"] = False
        else:
            try:
                measurement_metadata["source_column"] = resolve_energy_column(frame, energy_config)
                measurement_metadata["config"] = energy_config.to_dict()
                measurement_metadata["research_ready"] = True
            except Exception as exc:
                issues.append(
                    AuditIssue(
                        code="energy_unit_configuration_invalid",
                        severity="error",
                        message=str(exc),
                    )
                )
                measurement_metadata["research_ready"] = False
    if missing_rows:
        issues.append(
            AuditIssue(
                code="missing_timestamp",
                severity="error",
                message="Rows without a parseable timestamp were excluded from temporal checks.",
                count=missing_rows,
            )
        )
    if duplicate_count:
        issues.append(
            AuditIssue(
                code="duplicate_timestamp",
                severity="warning",
                message="Duplicate timestamps require an explicit aggregation policy.",
                count=duplicate_count,
            )
        )

    deltas = valid_timestamp.diff().dropna().dt.total_seconds().div(60)
    inferred = float(deltas.median()) if not deltas.empty else None
    if (
        expected is not None
        and inferred is not None
        and abs(inferred - expected) > max(1, expected * 0.1)
    ):
        issues.append(
            AuditIssue(
                code="unexpected_sampling_interval",
                severity="warning",
                message=f"Median interval {inferred:g} minutes differs from expected {expected}.",
            )
        )

    gap_count = 0
    if expected and not deltas.empty:
        gap_count = int((deltas > expected * 1.5).sum())
        if gap_count:
            issues.append(
                AuditIssue(
                    code="missing_intervals",
                    severity="warning",
                    message="One or more timestamp gaps exceed the expected sampling interval.",
                    count=gap_count,
                )
            )

    missing_percentages = {
        str(column): round(float(canonical[column].isna().mean() * 100), 4)
        for column in canonical.columns
    }
    invalid: dict[str, int] = {}

    def add_invalid(column: str, mask: pd.Series, message: str) -> None:
        count = int(mask.fillna(False).sum())
        if count:
            invalid[column] = count
            issues.append(
                AuditIssue(
                    code=f"invalid_{column}", severity="warning", message=message, count=count
                )
            )

    energy_values = _numeric_column(canonical, "energy_wh")
    if not energy_values.notna().any():
        for candidate in ("energy_kwh", "power_w", "power_kw", "energy", "consumption"):
            candidate_values = _numeric_column(canonical, candidate)
            if candidate_values.notna().any():
                energy_values = candidate_values
                break
    add_invalid(
        "energy_wh",
        energy_values < 0,
        "Energy/power values are negative.",
    )
    add_invalid(
        "occupancy", _numeric_column(canonical, "occupancy") < 0, "Occupancy values are negative."
    )
    add_invalid(
        "temperature_c",
        _numeric_column(canonical, "temperature_c").notna()
        & ~_numeric_column(canonical, "temperature_c").between(-60, 70),
        "Temperature is outside the broad physical plausibility range [-60, 70] °C.",
    )
    add_invalid(
        "humidity_pct",
        _numeric_column(canonical, "humidity_pct").notna()
        & ~_numeric_column(canonical, "humidity_pct").between(0, 100),
        "Relative humidity is outside [0, 100] percent.",
    )

    building_values = (
        canonical["building_id"].dropna().astype(str).drop_duplicates().sort_values().tolist()
        if "building_id" in canonical
        else []
    )
    if not building_values and building_id is not None:
        building_values = [str(building_id)]
    report = DataQualityReport(
        source_name=source_name,
        source_kind=source_kind,
        rows=len(canonical),
        columns=[str(column) for column in canonical.columns],
        timestamp_start=_safe_iso(valid_timestamp.iloc[0]) if len(valid_timestamp) else None,
        timestamp_end=_safe_iso(valid_timestamp.iloc[-1]) if len(valid_timestamp) else None,
        inferred_interval_minutes=inferred,
        expected_interval_minutes=expected,
        duplicate_timestamps=duplicate_count,
        missing_timestamp_rows=missing_rows,
        missing_percentages=missing_percentages,
        invalid_value_counts=invalid,
        buildings=building_values,
        issues=issues,
        metadata={
            "gap_count": gap_count,
            "timezone": timezone,
            "measurement": measurement_metadata,
            "unit_note": {
                "energy": "Energy/power units are retained from source labels; no magnitude-based conversion is inferred.",
                "occupancy": "Occupancy is treated as a non-negative count.",
                "weather": "Temperature is expected in degrees Celsius and humidity in percent when those labels are present.",
            }.get(source_kind, "Units require source documentation."),
        },
    )
    return report


def _merge_coverage(reports: list[DataQualityReport]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    for report in reports:
        bucket = by_kind.setdefault(
            report.source_kind,
            {"start": None, "end": None, "files": 0, "buildings": set(), "rows": 0},
        )
        bucket["files"] += 1
        bucket["rows"] += report.rows
        bucket["buildings"].update(report.buildings)
        if report.timestamp_start and (
            bucket["start"] is None or report.timestamp_start < bucket["start"]
        ):
            bucket["start"] = report.timestamp_start
        if report.timestamp_end and (bucket["end"] is None or report.timestamp_end > bucket["end"]):
            bucket["end"] = report.timestamp_end
    for bucket in by_kind.values():
        bucket["buildings"] = sorted(bucket["buildings"])
    return by_kind


def audit_directory(
    root: str | Path,
    *,
    timezone: str = "Asia/Kolkata",
    max_file_bytes: int | None = None,
    energy_config: EnergyMeasurementConfig | None = None,
) -> dict[str, Any]:
    """Audit every supported table and summarize usable modality coverage."""

    discovered = discover_files(root)
    reports: list[DataQualityReport] = []
    failures: list[dict[str, str]] = []
    for kind, paths in discovered.items():
        if kind == "unknown":
            continue
        for path in paths:
            try:
                frame = read_table(path, max_bytes=max_file_bytes)
                report = audit_frame(
                    frame,
                    source_name=str(path),
                    source_kind=kind,
                    timezone=timezone,
                    energy_config=energy_config if kind == "energy" else None,
                    building_id=(
                        _infer_building_from_path(path)
                        if kind in {"energy", "occupancy"}
                        else None
                    ),
                )
                reports.append(report)
            except (
                Exception
            ) as exc:  # audit should explain bad files rather than stop at first file
                failures.append({"path": str(path), "kind": kind, "error": str(exc)})

    energy_reports = [report for report in reports if report.source_kind == "energy"]
    energy_ready = bool(energy_reports) and all(
        report.metadata.get("measurement", {}).get("research_ready", False)
        for report in energy_reports
    )
    return {
        "audit_version": "0.1",
        "root": str(Path(root)),
        "discovered_files": {
            kind: [str(path) for path in paths]
            for kind, paths in discovered.items()
            if kind != "unknown"
        },
        "unknown_files": [str(path) for path in discovered.get("unknown", [])],
        "reports": [report.to_dict() for report in reports],
        "coverage_by_kind": _merge_coverage(reports),
        "failures": failures,
        "weather_note": (
            "Weather coverage is reported as observed source coverage. No full-period weather values "
            "are assumed or synthesized."
        ),
        "research_ready": energy_ready,
        "research_readiness_note": (
            "Energy unit and semantics configuration is required before experiments."
            if not energy_ready
            else "Energy unit and semantics configuration was supplied for all audited energy files."
        ),
    }
