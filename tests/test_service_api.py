from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import campus_forecast.services as services_module
from campus_forecast.api.main import app
from campus_forecast.data.audit import audit_directory
from campus_forecast.data.loader import discover_files
from campus_forecast.evaluation.robustness import CorruptionConfig
from campus_forecast.schema import (
    EnergyMeasurementConfig,
    FeatureConfiguration,
    ImputationStrategy,
    MeasurementSemantics,
    MeasurementUnit,
)
from campus_forecast.services import (
    run_occupancy_experiment,
    run_probabilistic_experiment,
    run_robustness_experiment,
    run_training_experiment,
    run_transfer_experiment,
)

TEST_ENERGY_CONFIG = EnergyMeasurementConfig(
    source_unit=MeasurementUnit.W,
    semantics=MeasurementSemantics.INSTANTANEOUS_POWER,
    expected_interval_minutes=10,
)


def _write_dataset(root) -> None:
    timestamps = pd.date_range("2024-01-01", periods=10 * 144, freq="10min", tz="UTC")
    energy = pd.DataFrame(
        {
            "timestamp": timestamps,
            "Power (watts)": 200 + np.sin(np.arange(len(timestamps)) / 20),
        }
    )
    occupancy = pd.DataFrame(
        {
            "timestamp": timestamps,
            "Occupancy": 20 + np.arange(len(timestamps)) % 10,
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": timestamps[::3],
            "Temperature": 25 + np.sin(np.arange(len(timestamps[::3])) / 10),
            "Humidity": 55,
        }
    )
    calendar = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "working_day": [
                1 if day.dayofweek < 5 else 0 for day in pd.date_range("2024-01-01", periods=10)
            ],
            "high_activity": 1,
        }
    )
    (root / "energy").mkdir()
    (root / "occupancy").mkdir()
    energy.to_csv(root / "energy" / "Academic.csv", index=False)
    occupancy.to_csv(root / "occupancy" / "Academic.csv", index=False)
    energy.assign(**{"Power (watts)": energy["Power (watts)"] + 35}).to_csv(
        root / "energy" / "Library.csv", index=False
    )
    occupancy.to_csv(root / "occupancy" / "Library.csv", index=False)
    weather.to_csv(root / "weather.csv", index=False)
    calendar.to_csv(root / "calendar.csv", index=False)


def test_directory_discovery_uses_modality_directories(tmp_path) -> None:
    _write_dataset(tmp_path)
    discovered = discover_files(tmp_path)
    assert discovered["energy"][0].name == "Academic.csv"
    assert discovered["occupancy"][0].name == "Academic.csv"
    audit = audit_directory(tmp_path)
    assert audit["coverage_by_kind"]["energy"]["rows"] > 0
    assert audit["coverage_by_kind"]["energy"]["buildings"] == ["academic", "library"]


def test_real_service_path_persists_actual_run_metadata(tmp_path) -> None:
    _write_dataset(tmp_path)
    result = run_training_experiment(
        root=tmp_path,
        dataset_id="fixture_dataset",
        dataset_version="fixture-v1",
        building_id="academic",
        model_name="linear_regression",
        horizon_name="1h",
        lookback_hours=1,
        feature_configuration=FeatureConfiguration.ENERGY_CALENDAR,
        random_seed=3,
        energy_config=TEST_ENERGY_CONFIG,
        experiment_root=tmp_path / "experiments",
    )
    assert result["status"] == "completed"
    assert result["data_is_demo"] is False
    assert "horizon_1_mae" in result["metrics"]


def test_api_health_and_path_validation() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        openapi = client.get("/openapi.json").json()
        assert "/api/energy/predict" in openapi["paths"]
        assert "/api/scenarios" in openapi["paths"]
        rejected = client.post(
            "/api/datasets/audit",
            json={"root_path": "C:/outside", "dataset_id": "outside", "dataset_version": "x"},
        )
        assert rejected.status_code == 400
        missing_measurement = client.post(
            "/api/energy/predict",
            json={"dataset_id": "not_registered", "building_id": "academic"},
        )
        assert missing_measurement.status_code == 422
        assert "EnergyMeasurementConfig" in missing_measurement.json()["detail"]


def test_probabilistic_service_calibrates_on_validation_only(tmp_path) -> None:
    _write_dataset(tmp_path)
    result = run_probabilistic_experiment(
        root=tmp_path,
        dataset_id="fixture_probabilistic",
        dataset_version="fixture-v1",
        building_id="academic",
        horizon_name="10m",
        lookback_hours=1,
        feature_configuration=FeatureConfiguration.ENERGY_CALENDAR,
        random_seed=4,
        energy_config=TEST_ENERGY_CONFIG,
        experiment_root=tmp_path / "experiments",
    )
    assert result["model_name"] == "quantile_direct_conformal"
    assert result["model_architecture"]["calibration"] == "split_conformal_on_validation_only"
    assert "horizon_1_interval_coverage_80" in result["metrics"]


def test_occupancy_stage_is_evaluated_separately(tmp_path) -> None:
    _write_dataset(tmp_path)
    result = run_occupancy_experiment(
        root=tmp_path,
        dataset_id="fixture_occupancy",
        dataset_version="fixture-v1",
        building_id="academic",
        horizon_name="1h",
        random_seed=8,
        experiment_root=tmp_path / "experiments",
    )
    assert result["model_name"] == "seasonal_naive_occupancy"
    assert "horizon_1_mae" in result["metrics"]


def test_tree_occupancy_service_is_evaluated_out_of_sample(tmp_path) -> None:
    _write_dataset(tmp_path)
    result = run_occupancy_experiment(
        root=tmp_path,
        dataset_id="fixture_occupancy_tree",
        dataset_version="fixture-v1",
        building_id="academic",
        horizon_name="1h",
        random_seed=9,
        model_name="tree",
        lookback_hours=1,
        experiment_root=tmp_path / "experiments",
    )
    assert result["model_name"] == "tree_occupancy"
    assert result["model_architecture"]["family"] == "hist_gradient_boosting"
    assert "horizon_6_mae" in result["metrics"]


def test_decode_lstm_service_wiring_uses_validation_and_sequence_output(
    tmp_path, monkeypatch
) -> None:
    _write_dataset(tmp_path)

    class FakeSequenceModel:
        cell = "lstm"
        hidden_size = 32
        dense_units = 5
        history = [{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.1}]
        checkpoint_path = None

        def fit(self, features, targets, *, validation=None):
            assert features.ndim == 3
            assert validation is not None
            assert validation[0].ndim == 3
            self.horizon_steps = targets.shape[1]
            return self

        def predict(self, features):
            return np.zeros((len(features), self.horizon_steps), dtype=float)

    monkeypatch.setattr(
        services_module,
        "_model_for_name",
        lambda name, random_seed: (FakeSequenceModel(), "decode_lstm"),
    )
    result = run_training_experiment(
        root=tmp_path,
        dataset_id="fixture_lstm_wiring",
        dataset_version="fixture-v1",
        building_id="academic",
        model_name="decode_lstm",
        horizon_name="10m",
        lookback_hours=1,
        feature_configuration=FeatureConfiguration.ENERGY_CALENDAR,
        random_seed=10,
        energy_config=TEST_ENERGY_CONFIG,
        experiment_root=tmp_path / "experiments",
    )
    assert result["model_name"] == "decode_lstm"
    assert result["model_architecture"]["family"] == "pytorch_sequence"
    assert result["model_architecture"]["validation_checkpoint_selection"] is True


def test_robustness_service_records_clean_and_corrupted_metrics(tmp_path) -> None:
    _write_dataset(tmp_path)
    result = run_robustness_experiment(
        root=tmp_path,
        dataset_id="fixture_robustness",
        dataset_version="fixture-v1",
        building_id="academic",
        model_name="linear_regression",
        horizon_name="10m",
        lookback_hours=1,
        feature_configuration=FeatureConfiguration.ENERGY_WEATHER_CALENDAR,
        corruption=CorruptionConfig(seed=5, weather_missing_percent=20),
        imputation_strategy=ImputationStrategy.CAUSAL,
        random_seed=5,
        energy_config=TEST_ENERGY_CONFIG,
        experiment_root=tmp_path / "experiments",
    )
    assert "clean_horizon_1_mae" in result["metrics"]
    assert "corrupted_horizon_1_mae" in result["metrics"]
    assert "horizon_1_mae_degradation_percent" in result["metrics"]


def test_transfer_service_runs_zero_shot_and_few_shot_plan(tmp_path) -> None:
    _write_dataset(tmp_path)
    result = run_transfer_experiment(
        root=tmp_path,
        dataset_id="fixture_transfer",
        dataset_version="fixture-v1",
        target_building="academic",
        model_name="linear_regression",
        horizon_name="10m",
        lookback_hours=1,
        feature_configuration=FeatureConfiguration.ENERGY_CALENDAR,
        adaptation_days=(1, 3),
        random_seed=6,
        energy_config=TEST_ENERGY_CONFIG,
        experiment_root=tmp_path / "experiments",
    )
    assert result["model_name"] == "transfer_linear_regression"
    assert any(row["setting"] == "zero_shot" for row in result["transfer_rows"])
