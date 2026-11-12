from __future__ import annotations

import pandas as pd

from campus_forecast.evaluation.robustness import CorruptionConfig, corrupt_frame
from campus_forecast.evaluation.transfer import plan_adaptation_splits
from campus_forecast.pipeline.splits import leave_one_building_out


def test_corruption_is_reproducible(regular_frame: pd.DataFrame) -> None:
    config = CorruptionConfig(seed=7, occupancy_missing_percent=20, outage_hours=6)
    first = corrupt_frame(regular_frame, config)
    second = corrupt_frame(regular_frame, config)
    pd.testing.assert_frame_equal(first, second)
    assert first["occupancy"].isna().sum() > 0


def test_few_shot_adaptation_splits_do_not_overlap(regular_frame: pd.DataFrame) -> None:
    plans = plan_adaptation_splits(regular_frame, target_building="A", adaptation_days=(1, 3))
    assert len(plans) == 2
    for plan in plans:
        assert pd.Timestamp(plan.adaptation_end) < pd.Timestamp(plan.evaluation_start)


def test_leave_one_building_out_excludes_target_from_training(regular_frame: pd.DataFrame) -> None:
    folds = leave_one_building_out(regular_frame)
    assert {fold.target_building for fold in folds} == {"A", "B"}
    for fold in folds:
        assert fold.target_building not in fold.training_buildings
        assert set(fold.training_buildings) == {"A", "B"} - {fold.target_building}
