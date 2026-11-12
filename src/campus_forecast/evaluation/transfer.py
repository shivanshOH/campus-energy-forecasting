"""Cross-building generalization and few-shot adaptation planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from campus_forecast.exceptions import DataValidationError, LeakageError
from campus_forecast.pipeline.splits import leave_one_building_out


@dataclass(frozen=True)
class AdaptationSplit:
    target_building: str
    days: int
    adaptation_start: str
    adaptation_end: str
    evaluation_start: str
    evaluation_end: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def plan_adaptation_splits(
    frame: pd.DataFrame,
    *,
    target_building: str,
    adaptation_days: tuple[int, ...] = (1, 3, 7, 14, 30),
) -> list[AdaptationSplit]:
    required = {"timestamp", "building_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Transfer frame is missing columns: {missing}")
    target = frame.loc[frame["building_id"].astype(str) == str(target_building)].copy()
    if target.empty:
        raise DataValidationError(f"Target building not found: {target_building}")
    target["timestamp"] = pd.to_datetime(target["timestamp"], utc=True)
    target = target.sort_values("timestamp")
    start = target["timestamp"].iloc[0]
    end = target["timestamp"].iloc[-1]
    plans: list[AdaptationSplit] = []
    for days in adaptation_days:
        if days < 1:
            raise ValueError("adaptation days must be positive")
        adaptation_end = start + pd.Timedelta(days=days) - pd.Timedelta(minutes=10)
        evaluation_start = adaptation_end + pd.Timedelta(minutes=10)
        if evaluation_start > end:
            continue
        plans.append(
            AdaptationSplit(
                target_building=str(target_building),
                days=days,
                adaptation_start=start.isoformat(),
                adaptation_end=min(adaptation_end, end).isoformat(),
                evaluation_start=evaluation_start.isoformat(),
                evaluation_end=end.isoformat(),
            )
        )
    return plans


def build_transfer_plan(frame: pd.DataFrame) -> dict[str, object]:
    """Return seven-fold zero-shot and few-shot split metadata."""

    folds = leave_one_building_out(frame)
    result: dict[str, object] = {"folds": []}
    for fold in folds:
        plans = plan_adaptation_splits(frame, target_building=fold.target_building)
        result["folds"].append(
            {
                "target_building": fold.target_building,
                "training_buildings": list(fold.training_buildings),
                "few_shot": [plan.to_dict() for plan in plans],
            }
        )
    return result


def assert_adaptation_does_not_touch_evaluation(
    adaptation_end: object, evaluation_start: object
) -> None:
    if pd.Timestamp(adaptation_end) >= pd.Timestamp(evaluation_start):
        raise LeakageError("Few-shot adaptation period overlaps evaluation period")


def transferability_matrix(rows: list[dict[str, object]], *, metric: str = "mae") -> pd.DataFrame:
    """Pivot saved fold metrics into a building transferability matrix."""

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    required = {"source_building", "target_building", metric}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Transfer rows are missing columns: {missing}")
    return frame.pivot_table(
        index="source_building", columns="target_building", values=metric, aggfunc="mean"
    )
