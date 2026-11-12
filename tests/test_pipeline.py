from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campus_forecast.exceptions import LeakageError
from campus_forecast.models.occupancy import (
    SeasonalNaiveOccupancyForecaster,
    TreeOccupancyForecaster,
)
from campus_forecast.pipeline.features import make_issue_table
from campus_forecast.pipeline.preprocessing import CausalImputer, TrainOnlyMinMaxScaler
from campus_forecast.pipeline.splits import chronological_split
from campus_forecast.pipeline.windows import create_windows
from campus_forecast.schema import FeatureConfiguration, ImputationStrategy


def test_chronological_split_is_disjoint(regular_frame: pd.DataFrame) -> None:
    one_building = regular_frame.loc[regular_frame["building_id"] == "A"]
    train, validation, test, boundaries = chronological_split(one_building)
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    assert train["timestamp"].max() == boundaries.train_end


def test_scaler_uses_training_statistics_only() -> None:
    train = pd.DataFrame({"value": [0.0, 10.0]})
    test = pd.DataFrame({"value": [20.0]})
    scaler = TrainOnlyMinMaxScaler(columns=["value"]).fit(train)
    transformed = scaler.transform(test)
    assert transformed.iloc[0, 0] == 2.0


def test_causal_imputer_carries_training_state_not_test_future() -> None:
    train = pd.DataFrame({"occupancy": [5.0, 7.0]})
    test = pd.DataFrame({"occupancy": [np.nan, np.nan, 9.0]})
    imputer = CausalImputer(strategy=ImputationStrategy.CAUSAL, columns=["occupancy"])
    imputer.fit(train)
    transformed = imputer.transform(test)
    assert transformed["occupancy"].tolist() == [7.0, 7.0, 9.0]


def test_offline_imputation_is_rejected_without_explicit_flag() -> None:
    from campus_forecast.evaluation.robustness import impute_frame

    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=2, freq="10min"), "occupancy": [1, None]}
    )
    with pytest.raises(LeakageError):
        impute_frame(frame, strategy=ImputationStrategy.TIME_INTERPOLATION_OFFLINE)


def test_windows_have_correct_direct_horizon_and_no_leakage(regular_frame: pd.DataFrame) -> None:
    one_building = regular_frame.loc[regular_frame["building_id"] == "A"].head(20)
    windows = create_windows(
        one_building,
        feature_columns=["energy_wh", "occupancy"],
        lookback_steps=4,
        horizon_steps=3,
        interval_minutes=10,
    )
    assert windows.features.shape == (14, 4, 2)
    assert windows.targets.shape == (14, 3)
    assert (windows.target_timestamps[:, 0] > windows.issue_timestamps.to_numpy()).all()
    assert windows.targets[0, 0] == one_building.iloc[4]["energy_wh"]


def test_oracle_requires_explicit_configuration(regular_frame: pd.DataFrame) -> None:
    with pytest.raises(LeakageError):
        make_issue_table(
            regular_frame.loc[regular_frame["building_id"] == "A"].head(200),
            configuration=FeatureConfiguration.HISTORICAL_OCCUPANCY,
            lookback_steps=4,
            horizon_steps=1,
            occupancy_mode="oracle",  # type: ignore[arg-type]
        )


def test_issue_time_occupancy_forecast_is_unchanged_by_future_observation() -> None:
    timestamps = pd.date_range("2024-01-01", periods=500, freq="10min", tz="UTC")
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "building_id": "A",
            "occupancy": np.arange(len(timestamps), dtype=float),
        }
    )
    issue = pd.DataFrame({"timestamp": [timestamps[300]], "building_id": ["A"]})
    first = SeasonalNaiveOccupancyForecaster().fit(base).predict(issue, horizon_steps=6)
    modified = base.copy()
    modified.loc[modified["timestamp"] == timestamps[303], "occupancy"] = 99999.0
    second = SeasonalNaiveOccupancyForecaster().fit(modified).predict(issue, horizon_steps=6)
    pd.testing.assert_series_equal(
        first["occupancy_pred"].reset_index(drop=True),
        second["occupancy_pred"].reset_index(drop=True),
    )


def test_tree_occupancy_model_trains_chronologically_and_predicts_nonnegative(
    regular_frame: pd.DataFrame,
) -> None:
    occupancy = regular_frame[["timestamp", "building_id", "occupancy"]].copy()
    train = occupancy.loc[
        occupancy["timestamp"] < occupancy["timestamp"].min() + pd.Timedelta(days=2)
    ]
    model = TreeOccupancyForecaster(lookback_steps=6).fit(train, horizon_steps=3)
    issue_rows = occupancy.loc[
        occupancy["timestamp"] >= occupancy["timestamp"].min() + pd.Timedelta(days=2)
    ][["timestamp", "building_id"]].head(3)
    predictions = model.predict(occupancy, issue_rows, horizon_steps=3)
    assert len(predictions) == 9
    assert (predictions["occupancy_pred"] >= 0).all()
