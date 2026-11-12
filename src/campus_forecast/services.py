"""Application services shared by the API and command-line workflows."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campus_forecast.config import get_settings
from campus_forecast.data.loader import IBlendLoader, discover_files, merge_modalities
from campus_forecast.evaluation.experiments import ExperimentManager
from campus_forecast.evaluation.metrics import evaluate_point, evaluate_probabilistic
from campus_forecast.evaluation.robustness import (
    CorruptionConfig,
    corrupt_frame,
    degradation_percent,
    impute_frame,
)
from campus_forecast.evaluation.transfer import transferability_matrix
from campus_forecast.exceptions import DataUnavailableError, DataValidationError
from campus_forecast.models.baselines import (
    sklearn_model_suite,
)
from campus_forecast.models.energy import DecodeLSTMRegressor, GRURegressor
from campus_forecast.models.occupancy import (
    SeasonalNaiveOccupancyForecaster,
    TreeOccupancyForecaster,
)
from campus_forecast.models.probabilistic import (
    ConformalIntervalCalibrator,
    QuantileDirectForecaster,
)
from campus_forecast.pipeline.features import (
    build_calendar_features,
    feature_columns_for,
    make_issue_table,
)
from campus_forecast.pipeline.preprocessing import CausalImputer
from campus_forecast.pipeline.splits import chronological_split
from campus_forecast.schema import (
    HORIZONS,
    EnergyMeasurementConfig,
    ExperimentRecord,
    FeatureConfiguration,
    ImputationStrategy,
)
from campus_forecast.utils import stable_hash

BUILDING_CATEGORIES = {
    "academic": "academic",
    "acb": "academic",
    "lecture": "academic",
    "lcb": "academic",
    "library": "library",
    "lib": "library",
    "dining": "dining",
    "db": "dining",
    "facilities": "facilities",
    "fb": "facilities",
    "boys": "hostel",
    "boys_hostel": "hostel",
    "bh": "hostel",
    "girls": "hostel",
    "girls_hostel": "hostel",
    "gh": "hostel",
}


def _infer_building(path: Path) -> str:
    name = path.stem.strip().lower().replace(" ", "_")
    for token in ("energy", "power", "occupancy", "weather", "calendar"):
        name = name.replace(token, "")
    name = name.strip("_-.")
    return name or path.stem


def load_dataset_directory(
    root: str | Path,
    *,
    timezone: str = "Asia/Kolkata",
    energy_config: EnergyMeasurementConfig | None = None,
    require_energy: bool = True,
) -> dict[str, pd.DataFrame | None]:
    """Load a conventional I-BLEND directory without inventing absent modalities."""

    discovered = discover_files(root)
    loader = IBlendLoader(timezone=timezone)

    def load_many(kind: str) -> pd.DataFrame | None:
        frames: list[pd.DataFrame] = []
        for path in discovered.get(kind, []):
            building = _infer_building(path) if kind in {"energy", "occupancy"} else None
            try:
                frames.append(
                    loader.load(
                        path,
                        source_kind=kind,
                        building_id=building,
                        energy_config=energy_config,
                    )
                )
            except DataValidationError:
                # Re-raise with a source-specific path to avoid hiding malformed files.
                raise DataValidationError(f"Could not load {kind} source {path}")
        if not frames:
            return None
        combined = (
            pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        )
        combined.attrs.update(frames[0].attrs)
        return combined

    energy = load_many("energy") if require_energy else None
    if require_energy and energy is None:
        raise DataUnavailableError("No energy source files were found in the dataset root")
    occupancy = load_many("occupancy")
    weather = load_many("weather")
    calendar = load_many("calendar")
    if weather is not None and "building_id" not in weather:
        weather["building_id"] = "campus"
    return {"energy": energy, "occupancy": occupancy, "weather": weather, "calendar": calendar}


def _occupancy_prediction_table(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    horizon_steps: int,
    model_name: str = "seasonal_naive",
    lookback_steps: int = 144,
    seasonal_steps: int = 144,
    interval_minutes: int = 10,
) -> pd.DataFrame:
    """Create occupancy predictions keyed by issue and target timestamp."""

    if "occupancy" not in frame:
        raise DataValidationError("Occupancy predictions require an occupancy column")
    output = frame.copy().sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    issue_rows = output[["timestamp", "building_id"]].drop_duplicates()
    model_key = model_name.lower().replace("-", "_")
    if model_key in {"seasonal_naive", "seasonal", "naive"}:
        forecaster = SeasonalNaiveOccupancyForecaster(
            seasonal_steps=seasonal_steps,
            interval_minutes=interval_minutes,
        )
        # Seasonal naive has no learned parameters. Retain the complete
        # history so issue-time predictions can use observations that became
        # available during validation/test, while predict() filters every
        # lookup to the issue timestamp.
        forecaster.fit(output)
        return forecaster.predict(issue_rows, horizon_steps=horizon_steps)
    if model_key in {"tree", "hist_gradient_boosting", "lightgbm"}:
        train_frame = output.loc[output["timestamp"] <= train_end].copy()
        forecaster = TreeOccupancyForecaster(
            lookback_steps=lookback_steps,
        )
        forecaster.fit(train_frame, horizon_steps=horizon_steps)
        minimum_issue_time = pd.Timestamp(output["timestamp"].min()) + pd.Timedelta(
            minutes=interval_minutes * lookback_steps
        )
        issue_rows = issue_rows.loc[issue_rows["timestamp"] >= minimum_issue_time]
        return forecaster.predict(output, issue_rows, horizon_steps=horizon_steps)
    raise DataValidationError(f"Unknown occupancy model {model_name!r}; use seasonal_naive or tree")


def attach_seasonal_predicted_occupancy(
    frame: pd.DataFrame,
    *,
    seasonal_steps: int = 144,
    interval_minutes: int = 10,
) -> pd.DataFrame:
    """Attach a one-step forecast issued one interval before each timestamp.

    This compatibility helper never reads occupancy after the issue timestamp.
    Multi-horizon energy experiments use the issue-keyed table returned by
    ``_occupancy_prediction_table`` instead of a target-only column.
    """

    output = frame.copy().sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    issue_rows = output[["timestamp", "building_id"]].copy()
    issue_rows["timestamp"] = issue_rows["timestamp"] - pd.Timedelta(minutes=interval_minutes)
    forecaster = SeasonalNaiveOccupancyForecaster(
        seasonal_steps=seasonal_steps,
        interval_minutes=interval_minutes,
    )
    forecaster.fit(output)
    predictions = forecaster.predict(issue_rows, horizon_steps=1)
    predictions["timestamp"] = predictions["timestamp"].dt.tz_convert("UTC")
    predictions = predictions.rename(columns={"occupancy_pred": "occupancy_pred"})
    output = output.merge(
        predictions[["timestamp", "building_id", "occupancy_pred"]],
        on=["timestamp", "building_id"],
        how="left",
    )
    return output


def _model_for_name(name: str, random_seed: int):
    aliases = {
        "linear": "linear_regression",
        "lr": "linear_regression",
        "tree": "decision_tree",
        "dt": "decision_tree",
        "forest": "random_forest",
        "rf": "random_forest",
    }
    key = aliases.get(name.lower(), name.lower())
    if key in {"lstm", "decode_lstm", "decode_style_lstm", "decode-style-lstm"}:
        return DecodeLSTMRegressor(seed=random_seed), "decode_lstm"
    if key in {"gru", "decode_gru"}:
        return GRURegressor(seed=random_seed), "gru"
    suite = sklearn_model_suite(random_state=random_seed)
    if key not in suite:
        raise DataValidationError(
            f"Unknown CPU baseline {name!r}; available: {sorted(suite)}. "
            "Use the PyTorch model module directly for DECODE-style LSTM training."
        )
    return suite[key], key


def _issue_table_to_sequences(
    table: Any,
    *,
    configuration: FeatureConfiguration,
    energy_column: str,
    lookback_steps: int,
) -> np.ndarray:
    """Recover temporal sequences from issue-table lag columns for PyTorch."""

    historical_columns = feature_columns_for(configuration, energy_column=energy_column)
    context_columns = sorted(
        column for column in table.features.columns if str(column).startswith("future_")
    )
    sequences: list[np.ndarray] = []
    for _, row in table.features.iterrows():
        history = np.asarray(
            [
                [float(row[f"{column}_lag_{lag}"]) for column in historical_columns]
                for lag in range(lookback_steps - 1, -1, -1)
            ],
            dtype=np.float32,
        )
        if context_columns:
            context = np.asarray(
                [float(row[column]) for column in context_columns], dtype=np.float32
            )
            history = np.concatenate(
                [history, np.repeat(context[None, :], lookback_steps, axis=0)], axis=1
            )
        sequences.append(history)
    feature_count = len(historical_columns) + len(context_columns)
    if not sequences:
        return np.empty((0, lookback_steps, feature_count), dtype=np.float32)
    return np.stack(sequences).astype(np.float32)


def _period_dict(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "start": pd.Timestamp(frame["timestamp"].min()).isoformat(),
        "end": pd.Timestamp(frame["timestamp"].max()).isoformat(),
    }


def run_occupancy_experiment(
    *,
    root: str | Path,
    dataset_id: str,
    dataset_version: str,
    building_id: str | None,
    horizon_name: str,
    random_seed: int,
    model_name: str = "seasonal_naive",
    lookback_hours: int = 24,
    experiment_root: str | Path = "artifacts/experiments",
) -> dict[str, Any]:
    """Evaluate the causal seasonal-naive occupancy stage independently."""

    if horizon_name not in HORIZONS:
        raise DataValidationError(f"Unknown horizon: {horizon_name}")
    sources = load_dataset_directory(root, require_energy=False)
    if sources["occupancy"] is None:
        raise DataUnavailableError("No occupancy source files were found")
    occupancy = sources["occupancy"].copy()
    if building_id is not None:
        occupancy = occupancy.loc[occupancy["building_id"].astype(str) == str(building_id)].copy()
    if occupancy.empty:
        raise DataUnavailableError(f"No occupancy observations for building {building_id!r}")
    occupancy = occupancy.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    train, validation, test, boundaries = chronological_split(occupancy)
    horizon_steps = HORIZONS[horizon_name].steps
    issue_rows = test.iloc[:-horizon_steps] if len(test) > horizon_steps else test.iloc[0:0]
    if issue_rows.empty:
        raise DataUnavailableError("Occupancy test period is too short for the selected horizon")
    predictions = _occupancy_prediction_table(
        occupancy,
        train_end=pd.Timestamp(train["timestamp"].max()),
        horizon_steps=horizon_steps,
        model_name=model_name,
        lookback_steps=max(1, lookback_hours * 6),
    )
    predictions = predictions.loc[
        predictions["issue_timestamp"].isin(pd.to_datetime(issue_rows["timestamp"], utc=True))
    ]
    actual_lookup = occupancy.set_index(["building_id", "timestamp"])["occupancy"]
    actual_by_horizon: list[list[float]] = [[] for _ in range(horizon_steps)]
    predicted_by_horizon: list[list[float]] = [[] for _ in range(horizon_steps)]
    preview: list[dict[str, Any]] = []
    for issue_timestamp, issue_group in predictions.groupby("issue_timestamp", sort=True):
        horizons: list[dict[str, Any]] = []
        for _, forecast_row in issue_group.sort_values("horizon_step").iterrows():
            key = (forecast_row["building_id"], forecast_row["timestamp"])
            if key not in actual_lookup:
                continue
            actual_value = float(actual_lookup.loc[key])
            predicted_value = max(0.0, float(forecast_row["occupancy_pred"]))
            horizon_index = int(forecast_row["horizon_step"]) - 1
            actual_by_horizon[horizon_index].append(actual_value)
            predicted_by_horizon[horizon_index].append(predicted_value)
            horizons.append(
                {
                    "horizon_step": int(forecast_row["horizon_step"]),
                    "timestamp": str(forecast_row["timestamp"]),
                    "actual": actual_value,
                    "predicted": predicted_value,
                }
            )
        if len(preview) < 48 and horizons:
            preview.append({"issue_timestamp": str(issue_timestamp), "horizons": horizons})
    metrics = {
        f"horizon_{index + 1}_{name}": value
        for index, (actual_horizon, predicted_horizon) in enumerate(
            zip(actual_by_horizon, predicted_by_horizon, strict=True)
        )
        for name, value in evaluate_point(actual_horizon, predicted_horizon).items()
    }
    record = ExperimentRecord(
        experiment_id=stable_hash(
            {
                "dataset": dataset_id,
                "building": building_id,
                "model": f"{model_name}_occupancy",
                "horizon": horizon_name,
                "seed": random_seed,
            }
        ),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        building_id=building_id,
        train_period=_period_dict(train),
        validation_period=_period_dict(validation),
        test_period=_period_dict(test),
        model_name=f"{model_name}_occupancy",
        model_architecture={
            "family": "seasonal_naive"
            if model_name.lower().replace("-", "_") in {"seasonal_naive", "seasonal", "naive"}
            else "hist_gradient_boosting",
            "seasonal_steps": 144
            if model_name.lower().replace("-", "_") in {"seasonal_naive", "seasonal", "naive"}
            else None,
            "target": "occupancy",
            "forecast_preview": preview,
        },
        hyperparameters={"horizon": horizon_name, "lookback_hours": lookback_hours},
        feature_configuration="occupancy_forecast",
        input_window_steps=max(1, lookback_hours * 6),
        forecast_horizon_steps=horizon_steps,
        random_seed=random_seed,
        corruption_configuration={},
        target_unit="occupancy_count",
        measurement_metadata={
            "target_unit": "occupancy_count",
            "occupancy_model": model_name,
        },
        metrics=metrics,
        status="completed",
        data_is_demo=False,
    )
    manager = ExperimentManager(experiment_root)
    manager.save(record)
    return record.to_dict()


def _prepare_issue_table(
    *,
    root: str | Path,
    building_id: str | None,
    feature_configuration: FeatureConfiguration,
    horizon_name: str,
    lookback_hours: int,
    energy_config: EnergyMeasurementConfig | None = None,
    occupancy_model_name: str = "seasonal_naive",
) -> tuple[Any, pd.DataFrame, Any, int, int]:
    """Load, align and window one experiment without fitting a model."""

    sources = load_dataset_directory(root, energy_config=energy_config)
    occupancy_configurations = {
        FeatureConfiguration.HISTORICAL_OCCUPANCY,
        FeatureConfiguration.PREDICTED_OCCUPANCY,
        FeatureConfiguration.ORACLE_OCCUPANCY,
    }
    if sources["occupancy"] is None and feature_configuration in occupancy_configurations:
        raise DataUnavailableError(
            "The requested occupancy-aware configuration has no occupancy source"
        )
    merged = merge_modalities(
        sources["energy"],
        sources["occupancy"],
        sources["weather"],
        sources["calendar"],
        interval="10min",
        energy_config=energy_config,
    )
    if building_id is not None:
        merged = merged.loc[merged["building_id"].astype(str) == str(building_id)].copy()
    if merged.empty:
        raise DataUnavailableError(f"No aligned observations for building {building_id!r}")
    energy_column = str(merged.attrs.get("energy_column", "energy_wh"))
    _, _, _, boundaries = chronological_split(
        merged[["timestamp", energy_column]].drop_duplicates()
    )
    # I-BLEND weather is commonly coarser than the ten-minute grid. Causal
    # carry-forward is applied only after fitting each imputer on the training
    # period; it does not interpolate from future observations.
    imputation_columns = [
        column for column in ("occupancy", "temperature_c", "humidity_pct") if column in merged
    ]
    if imputation_columns:
        merged_attrs = merged.attrs.copy()
        imputed_pieces: list[pd.DataFrame] = []
        for _, group in merged.groupby("building_id", sort=False):
            train_group = group.loc[group["timestamp"] <= boundaries.train_end]
            imputer = CausalImputer(columns=imputation_columns)
            imputer.fit(train_group)
            imputed_pieces.append(imputer.transform(group))
        merged = (
            pd.concat(imputed_pieces, ignore_index=True)
            .sort_values(["building_id", "timestamp"])
            .reset_index(drop=True)
        )
        merged.attrs.update(merged_attrs)
    if feature_configuration in {
        FeatureConfiguration.PREDICTED_OCCUPANCY,
        FeatureConfiguration.ORACLE_OCCUPANCY,
    }:
        occupancy_predictions = _occupancy_prediction_table(
            merged,
            train_end=pd.Timestamp(boundaries.train_end),
            horizon_steps=HORIZONS[horizon_name].steps,
            model_name=occupancy_model_name,
            lookback_steps=max(1, lookback_hours * 6),
        )
    else:
        occupancy_predictions = None
    merged = build_calendar_features(merged)
    lookback_steps = max(1, lookback_hours * 6)
    horizon_steps = HORIZONS[horizon_name].steps
    table = make_issue_table(
        merged,
        configuration=feature_configuration,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        predicted_occupancy=occupancy_predictions,
        energy_column=energy_column,
    )
    if table.features.empty:
        raise DataUnavailableError("No complete, regular issue windows were available")
    target_end = pd.to_datetime(table.target_timestamps.iloc[:, -1], utc=True)
    train_mask = target_end <= boundaries.train_end
    validation_mask = (target_end > boundaries.train_end) & (
        target_end <= boundaries.validation_end
    )
    test_mask = (target_end > boundaries.validation_end) & (target_end <= boundaries.test_end)
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise DataUnavailableError(
            "The selected data has too few complete windows for all three splits"
        )
    return table, merged, boundaries, lookback_steps, horizon_steps


def run_training_experiment(
    *,
    root: str | Path,
    dataset_id: str,
    dataset_version: str,
    building_id: str | None,
    model_name: str,
    horizon_name: str,
    lookback_hours: int,
    feature_configuration: FeatureConfiguration,
    random_seed: int,
    energy_config: EnergyMeasurementConfig | None = None,
    occupancy_model_name: str = "seasonal_naive",
    experiment_root: str | Path = "artifacts/experiments",
) -> dict[str, Any]:
    """Train one direct multi-horizon experiment and persist honest metrics."""

    if horizon_name not in HORIZONS:
        raise DataValidationError(f"Unknown horizon: {horizon_name}")
    table, merged, boundaries, lookback_steps, horizon_steps = _prepare_issue_table(
        root=root,
        building_id=building_id,
        feature_configuration=feature_configuration,
        horizon_name=horizon_name,
        lookback_hours=lookback_hours,
        energy_config=energy_config,
        occupancy_model_name=occupancy_model_name,
    )
    energy_column = str(merged.attrs.get("energy_column", "energy_wh"))
    target_end = pd.to_datetime(table.target_timestamps.iloc[:, -1], utc=True)
    train_mask = target_end <= boundaries.train_end
    validation_mask = (target_end > boundaries.train_end) & (
        target_end <= boundaries.validation_end
    )
    test_mask = (target_end > boundaries.validation_end) & (target_end <= boundaries.test_end)

    model, canonical_model_name = _model_for_name(model_name, random_seed)
    started = time.perf_counter()
    is_sequence_model = canonical_model_name in {"decode_lstm", "gru"}
    checkpoint_suffix = "pt" if is_sequence_model else "joblib"
    checkpoint_path = (
        get_settings().checkpoints_dir
        / f"{canonical_model_name}-{stable_hash({dataset_id, building_id, horizon_name})}.{checkpoint_suffix}"
    )
    if is_sequence_model:
        sequence_features = _issue_table_to_sequences(
            table,
            configuration=feature_configuration,
            energy_column=energy_column,
            lookback_steps=lookback_steps,
        )
        model.checkpoint_path = checkpoint_path
        model.fit(
            sequence_features[train_mask.to_numpy()],
            table.targets.loc[train_mask].to_numpy(),
            validation=(
                sequence_features[validation_mask.to_numpy()],
                table.targets.loc[validation_mask].to_numpy(),
            ),
        )
        predictions = model.predict(sequence_features[test_mask.to_numpy()])
    else:
        model.fit(
            table.features.loc[train_mask].to_numpy(), table.targets.loc[train_mask].to_numpy()
        )
        predictions = model.predict(table.features.loc[test_mask].to_numpy())
    actual = table.targets.loc[test_mask].to_numpy()
    metrics: dict[str, float | None] = {}
    for index in range(horizon_steps):
        metrics.update(
            {
                f"horizon_{index + 1}_{name}": value
                for name, value in evaluate_point(actual[:, index], predictions[:, index]).items()
            }
        )
    point_preview = []
    test_issue_timestamps = table.issue_timestamps.loc[test_mask].tolist()
    test_target_timestamps = table.target_timestamps.loc[test_mask].to_numpy()
    for row_index in range(min(48, len(actual))):
        point_preview.append(
            {
                "issue_timestamp": str(test_issue_timestamps[row_index]),
                "horizons": [
                    {
                        "horizon_step": horizon + 1,
                        "timestamp": str(test_target_timestamps[row_index, horizon]),
                        "actual": float(actual[row_index, horizon]),
                        "p50": float(predictions[row_index, horizon]),
                    }
                    for horizon in range(horizon_steps)
                ],
            }
        )
    test_frame = merged.loc[
        (merged["timestamp"] > boundaries.validation_end)
        & (merged["timestamp"] <= boundaries.test_end)
    ]
    if not is_sequence_model:
        try:
            import joblib

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": model,
                    "feature_columns": list(table.features.columns),
                    "reference_features": table.features.loc[test_mask].head(1),
                },
                checkpoint_path,
            )
        except Exception:
            checkpoint_path = None
    model_architecture = {
        "family": "pytorch_sequence" if is_sequence_model else "sklearn_direct",
        "one_estimator_per_horizon": not is_sequence_model,
        "feature_columns": list(table.features.columns),
        "forecast_preview": point_preview,
    }
    if is_sequence_model:
        model_architecture.update(
            {
                "cell": model.cell,
                "hidden_size": model.hidden_size,
                "dense_units": model.dense_units,
                "optimizer": "RMSprop",
                "validation_checkpoint_selection": True,
                "training_history": model.history,
                "reproduction_note": (
                    "DECODE-style architecture; exact reproduction is not claimed where "
                    "window construction and unspecified preprocessing differ from the paper."
                ),
            }
        )
    hyperparameters: dict[str, Any] = {
        "lookback_hours": lookback_hours,
        "horizon": horizon_name,
    }
    if is_sequence_model:
        hyperparameters.update(
            {
                "hidden_size": model.hidden_size,
                "cell": model.cell,
                "dense_units": model.dense_units,
                "epochs": getattr(model, "epochs", None),
                "batch_size": getattr(model, "batch_size", None),
                "learning_rate": getattr(model, "learning_rate", None),
                "patience": getattr(model, "patience", None),
                "seed": getattr(model, "seed", random_seed),
            }
        )
    record = ExperimentRecord(
        experiment_id=stable_hash(
            {
                "dataset": dataset_id,
                "building": building_id,
                "model": canonical_model_name,
                "horizon": horizon_name,
                "features": feature_configuration.value,
                "seed": random_seed,
            }
        ),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        building_id=building_id,
        train_period=_period_dict(merged.loc[merged["timestamp"] <= boundaries.train_end]),
        validation_period=_period_dict(
            merged.loc[
                (merged["timestamp"] > boundaries.train_end)
                & (merged["timestamp"] <= boundaries.validation_end)
            ]
        ),
        test_period=_period_dict(test_frame),
        model_name=canonical_model_name,
        model_architecture=model_architecture,
        hyperparameters=hyperparameters,
        feature_configuration=feature_configuration.value,
        input_window_steps=lookback_steps,
        forecast_horizon_steps=horizon_steps,
        random_seed=random_seed,
        corruption_configuration={},
        target_unit=str(merged.attrs.get("energy_target_unit", "Wh")),
        measurement_metadata={
            "energy_column": energy_column,
            "energy_measurement_config": merged.attrs.get("energy_measurement_config", {}),
            "energy_conversion_summary": merged.attrs.get("energy_conversion_summary", {}),
            "occupancy_model": occupancy_model_name,
        },
        metrics=metrics,
        training_duration_seconds=time.perf_counter() - started,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        status="completed",
        data_is_demo=False,
    )
    manager = ExperimentManager(experiment_root)
    manager.save(record)
    return record.to_dict()


def run_probabilistic_experiment(
    *,
    root: str | Path,
    dataset_id: str,
    dataset_version: str,
    building_id: str | None,
    horizon_name: str,
    lookback_hours: int,
    feature_configuration: FeatureConfiguration,
    random_seed: int,
    energy_config: EnergyMeasurementConfig | None = None,
    occupancy_model_name: str = "seasonal_naive",
    experiment_root: str | Path = "artifacts/experiments",
) -> dict[str, Any]:
    """Train quantile models, calibrate on validation, and score untouched test data."""

    if horizon_name not in HORIZONS:
        raise DataValidationError(f"Unknown horizon: {horizon_name}")
    table, merged, boundaries, lookback_steps, horizon_steps = _prepare_issue_table(
        root=root,
        building_id=building_id,
        feature_configuration=feature_configuration,
        horizon_name=horizon_name,
        lookback_hours=lookback_hours,
        energy_config=energy_config,
        occupancy_model_name=occupancy_model_name,
    )
    target_end = pd.to_datetime(table.target_timestamps.iloc[:, -1], utc=True)
    train_mask = target_end <= boundaries.train_end
    validation_mask = (target_end > boundaries.train_end) & (
        target_end <= boundaries.validation_end
    )
    test_mask = (target_end > boundaries.validation_end) & (target_end <= boundaries.test_end)
    model = QuantileDirectForecaster(random_state=random_seed)
    started = time.perf_counter()
    model.fit(table.features.loc[train_mask].to_numpy(), table.targets.loc[train_mask].to_numpy())
    validation_prediction = model.predict(table.features.loc[validation_mask].to_numpy())
    test_prediction = model.predict(table.features.loc[test_mask].to_numpy())
    calibrated_lower: list[np.ndarray] = []
    calibrated_upper: list[np.ndarray] = []
    calibration_adjustments: list[float] = []
    metrics: dict[str, float | None] = {}
    for horizon in range(horizon_steps):
        calibrator = ConformalIntervalCalibrator(coverage_level=0.80).fit(
            table.targets.loc[validation_mask].to_numpy()[:, horizon],
            validation_prediction[0.1][:, horizon],
            validation_prediction[0.9][:, horizon],
        )
        lower, upper = calibrator.transform(
            test_prediction[0.1][:, horizon], test_prediction[0.9][:, horizon]
        )
        calibrated_lower.append(lower)
        calibrated_upper.append(upper)
        calibration_adjustments.append(float(calibrator.adjustment_ or 0.0))
        metrics.update(
            {
                f"horizon_{horizon + 1}_{name}": value
                for name, value in evaluate_probabilistic(
                    table.targets.loc[test_mask].to_numpy()[:, horizon],
                    lower,
                    test_prediction[0.5][:, horizon],
                    upper,
                ).items()
            }
        )
    probabilistic_preview = []
    test_issue_timestamps = table.issue_timestamps.loc[test_mask].tolist()
    test_target_timestamps = table.target_timestamps.loc[test_mask].to_numpy()
    test_actual = table.targets.loc[test_mask].to_numpy()
    for row_index in range(min(48, len(test_actual))):
        probabilistic_preview.append(
            {
                "issue_timestamp": str(test_issue_timestamps[row_index]),
                "horizons": [
                    {
                        "horizon_step": horizon + 1,
                        "timestamp": str(test_target_timestamps[row_index, horizon]),
                        "actual": float(test_actual[row_index, horizon]),
                        "p10": float(calibrated_lower[horizon][row_index]),
                        "p50": float(test_prediction[0.5][row_index, horizon]),
                        "p90": float(calibrated_upper[horizon][row_index]),
                    }
                    for horizon in range(horizon_steps)
                ],
            }
        )
    record = ExperimentRecord(
        experiment_id=stable_hash(
            {
                "dataset": dataset_id,
                "building": building_id,
                "model": "quantile_direct_conformal",
                "horizon": horizon_name,
                "features": feature_configuration.value,
                "seed": random_seed,
            }
        ),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        building_id=building_id,
        train_period=_period_dict(merged.loc[merged["timestamp"] <= boundaries.train_end]),
        validation_period=_period_dict(
            merged.loc[
                (merged["timestamp"] > boundaries.train_end)
                & (merged["timestamp"] <= boundaries.validation_end)
            ]
        ),
        test_period=_period_dict(
            merged.loc[
                (merged["timestamp"] > boundaries.validation_end)
                & (merged["timestamp"] <= boundaries.test_end)
            ]
        ),
        model_name="quantile_direct_conformal",
        model_architecture={
            "family": "quantile_regression",
            "quantiles": [0.1, 0.5, 0.9],
            "calibration": "split_conformal_on_validation_only",
            "calibration_adjustments": calibration_adjustments,
            "forecast_preview": probabilistic_preview,
        },
        hyperparameters={"lookback_hours": lookback_hours, "horizon": horizon_name},
        feature_configuration=feature_configuration.value,
        input_window_steps=lookback_steps,
        forecast_horizon_steps=horizon_steps,
        random_seed=random_seed,
        corruption_configuration={},
        target_unit=str(merged.attrs.get("energy_target_unit", "Wh")),
        measurement_metadata={
            "energy_column": str(merged.attrs.get("energy_column", "energy_wh")),
            "energy_measurement_config": merged.attrs.get("energy_measurement_config", {}),
            "energy_conversion_summary": merged.attrs.get("energy_conversion_summary", {}),
            "occupancy_model": occupancy_model_name,
        },
        metrics=metrics,
        training_duration_seconds=time.perf_counter() - started,
        status="completed",
        data_is_demo=False,
    )
    manager = ExperimentManager(experiment_root)
    manager.save(record)
    return record.to_dict()


def _causal_impute_by_building(
    corrupted: pd.DataFrame,
    clean_train: pd.DataFrame,
    *,
    strategy: ImputationStrategy,
) -> pd.DataFrame:
    if "building_id" not in corrupted:
        return impute_frame(
            corrupted,
            strategy=strategy,
            training_frame=clean_train,
        )
    pieces: list[pd.DataFrame] = []
    for building_id, group in corrupted.groupby("building_id", sort=False):
        train_group = clean_train.loc[clean_train["building_id"] == building_id]
        pieces.append(
            impute_frame(
                group,
                strategy=strategy,
                training_frame=train_group,
            )
        )
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["building_id", "timestamp"])
        .reset_index(drop=True)
    )


def run_robustness_experiment(
    *,
    root: str | Path,
    dataset_id: str,
    dataset_version: str,
    building_id: str | None,
    model_name: str,
    horizon_name: str,
    lookback_hours: int,
    feature_configuration: FeatureConfiguration,
    corruption: CorruptionConfig,
    imputation_strategy: ImputationStrategy,
    random_seed: int,
    energy_config: EnergyMeasurementConfig | None = None,
    occupancy_model_name: str = "seasonal_naive",
    experiment_root: str | Path = "artifacts/experiments",
) -> dict[str, Any]:
    """Score clean and corrupted data on the same leakage-safe test period."""

    table, merged, boundaries, lookback_steps, horizon_steps = _prepare_issue_table(
        root=root,
        building_id=building_id,
        feature_configuration=feature_configuration,
        horizon_name=horizon_name,
        lookback_hours=lookback_hours,
        energy_config=energy_config,
        occupancy_model_name=occupancy_model_name,
    )
    clean_train = merged.loc[merged["timestamp"] <= boundaries.train_end].copy()
    corrupted_frame = corrupt_frame(merged, corruption)
    corrupted_frame = _causal_impute_by_building(
        corrupted_frame,
        clean_train,
        strategy=imputation_strategy,
    )
    corrupted_frame = build_calendar_features(corrupted_frame)
    corrupted_energy_column = str(merged.attrs.get("energy_column", "energy_wh"))
    corrupted_occupancy_predictions = None
    if feature_configuration in {
        FeatureConfiguration.PREDICTED_OCCUPANCY,
        FeatureConfiguration.ORACLE_OCCUPANCY,
    }:
        corrupted_occupancy_predictions = _occupancy_prediction_table(
            corrupted_frame,
            train_end=pd.Timestamp(boundaries.train_end),
            horizon_steps=horizon_steps,
            model_name=occupancy_model_name,
            lookback_steps=lookback_steps,
        )
    corrupted_table = make_issue_table(
        corrupted_frame,
        configuration=feature_configuration,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        predicted_occupancy=corrupted_occupancy_predictions,
        energy_column=corrupted_energy_column,
    )
    target_end = pd.to_datetime(table.target_timestamps.iloc[:, -1], utc=True)
    train_mask = target_end <= boundaries.train_end
    clean_test_mask = (target_end > boundaries.validation_end) & (target_end <= boundaries.test_end)
    model, canonical_model_name = _model_for_name(model_name, random_seed)
    model.fit(table.features.loc[train_mask].to_numpy(), table.targets.loc[train_mask].to_numpy())
    clean_prediction = model.predict(table.features.loc[clean_test_mask].to_numpy())
    clean_actual = table.targets.loc[clean_test_mask].to_numpy()
    clean_metrics = {
        f"horizon_{index + 1}_{name}": value
        for index in range(horizon_steps)
        for name, value in evaluate_point(
            clean_actual[:, index], clean_prediction[:, index]
        ).items()
    }

    corrupted_end = pd.to_datetime(corrupted_table.target_timestamps.iloc[:, -1], utc=True)
    corrupted_test_mask = (corrupted_end > boundaries.validation_end) & (
        corrupted_end <= boundaries.test_end
    )
    if not corrupted_test_mask.any():
        raise DataUnavailableError("Corruption removed every complete test window")
    corrupted_prediction = model.predict(
        corrupted_table.features.loc[corrupted_test_mask].to_numpy()
    )
    corrupted_actual = corrupted_table.targets.loc[corrupted_test_mask].to_numpy()
    corrupted_metrics = {
        f"horizon_{index + 1}_{name}": value
        for index in range(horizon_steps)
        for name, value in evaluate_point(
            corrupted_actual[:, index], corrupted_prediction[:, index]
        ).items()
    }
    metrics = {f"clean_{key}": value for key, value in clean_metrics.items()}
    metrics.update({f"corrupted_{key}": value for key, value in corrupted_metrics.items()})
    for index in range(horizon_steps):
        clean_error = clean_metrics.get(f"horizon_{index + 1}_mae")
        corrupted_error = corrupted_metrics.get(f"horizon_{index + 1}_mae")
        metrics[f"horizon_{index + 1}_mae_degradation_percent"] = degradation_percent(
            clean_error, corrupted_error
        )
    record = ExperimentRecord(
        experiment_id=stable_hash(
            {
                "dataset": dataset_id,
                "building": building_id,
                "model": canonical_model_name,
                "robustness": corruption.to_dict(),
                "imputation": imputation_strategy.value,
            }
        ),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        building_id=building_id,
        train_period=_period_dict(clean_train),
        validation_period={
            "start": pd.Timestamp(boundaries.train_end).isoformat(),
            "end": pd.Timestamp(boundaries.validation_end).isoformat(),
        },
        test_period={
            "start": pd.Timestamp(boundaries.validation_end).isoformat(),
            "end": pd.Timestamp(boundaries.test_end).isoformat(),
        },
        model_name=f"robustness_{canonical_model_name}",
        model_architecture={"family": "clean_model_scored_on_corruption"},
        hyperparameters={"lookback_hours": lookback_hours, "horizon": horizon_name},
        feature_configuration=feature_configuration.value,
        input_window_steps=lookback_steps,
        forecast_horizon_steps=horizon_steps,
        random_seed=random_seed,
        corruption_configuration={
            **corruption.to_dict(),
            "imputation_strategy": imputation_strategy.value,
            "clean_test_rows": int(clean_test_mask.sum()),
            "corrupted_test_rows": int(corrupted_test_mask.sum()),
        },
        target_unit=str(merged.attrs.get("energy_target_unit", "Wh")),
        measurement_metadata={
            "energy_column": str(merged.attrs.get("energy_column", "energy_wh")),
            "energy_measurement_config": merged.attrs.get("energy_measurement_config", {}),
            "energy_conversion_summary": merged.attrs.get("energy_conversion_summary", {}),
            "occupancy_model": occupancy_model_name,
        },
        metrics=metrics,
        status="completed",
        data_is_demo=False,
    )
    manager = ExperimentManager(experiment_root)
    manager.save(record)
    return record.to_dict()


def run_transfer_experiment(
    *,
    root: str | Path,
    dataset_id: str,
    dataset_version: str,
    target_building: str,
    model_name: str,
    horizon_name: str,
    lookback_hours: int,
    feature_configuration: FeatureConfiguration,
    adaptation_days: tuple[int, ...],
    random_seed: int,
    energy_config: EnergyMeasurementConfig | None = None,
    occupancy_model_name: str = "seasonal_naive",
    experiment_root: str | Path = "artifacts/experiments",
) -> dict[str, Any]:
    """Run pooled zero-shot and simple few-shot refit evaluations."""

    table, merged, _, lookback_steps, horizon_steps = _prepare_issue_table(
        root=root,
        building_id=None,
        feature_configuration=feature_configuration,
        horizon_name=horizon_name,
        lookback_hours=lookback_hours,
        energy_config=energy_config,
        occupancy_model_name=occupancy_model_name,
    )
    target_end = pd.to_datetime(table.target_timestamps.iloc[:, -1], utc=True)
    building_ids = table.building_ids.astype(str).to_numpy()
    target_rows = building_ids == str(target_building)
    target_frame = merged.loc[merged["building_id"].astype(str) == str(target_building)]
    if target_frame.empty:
        raise DataUnavailableError(f"Target building not found: {target_building}")
    target_cutoff = pd.Timestamp(target_frame["timestamp"].iloc[int(len(target_frame) * 0.85)])
    source_train = (~target_rows) & (target_end <= target_cutoff)
    zero_test = target_rows & (target_end > target_cutoff)
    if not source_train.any() or not zero_test.any():
        raise DataUnavailableError("Insufficient rows for leave-one-building-out evaluation")
    model, canonical_model_name = _model_for_name(model_name, random_seed)
    model.fit(
        table.features.loc[source_train].to_numpy(), table.targets.loc[source_train].to_numpy()
    )
    zero_prediction = model.predict(table.features.loc[zero_test].to_numpy())
    zero_actual = table.targets.loc[zero_test].to_numpy()
    rows: list[dict[str, object]] = []
    for index in range(horizon_steps):
        rows.append(
            {
                "source_building": "pooled_six_buildings",
                "target_building": target_building,
                "setting": "zero_shot",
                "days": 0,
                **evaluate_point(zero_actual[:, index], zero_prediction[:, index]),
            }
        )
    target_start = pd.Timestamp(target_frame["timestamp"].min())
    adaptation_records: list[dict[str, object]] = []
    for days in adaptation_days:
        adaptation_end = target_start + pd.Timedelta(days=days)
        adaptation = target_rows & (target_end <= adaptation_end)
        evaluation = target_rows & (target_end > adaptation_end)
        if not adaptation.any() or not evaluation.any():
            continue
        adapted_model, _ = _model_for_name(model_name, random_seed)
        adapted_model.fit(
            pd.concat(
                [table.features.loc[source_train], table.features.loc[adaptation]],
                ignore_index=True,
            ).to_numpy(),
            pd.concat(
                [table.targets.loc[source_train], table.targets.loc[adaptation]],
                ignore_index=True,
            ).to_numpy(),
        )
        prediction = adapted_model.predict(table.features.loc[evaluation].to_numpy())
        actual = table.targets.loc[evaluation].to_numpy()
        for index in range(horizon_steps):
            adaptation_records.append(
                {
                    "source_building": "pooled_six_buildings",
                    "target_building": target_building,
                    "setting": "few_shot",
                    "days": days,
                    **evaluate_point(actual[:, index], prediction[:, index]),
                }
            )
    rows.extend(adaptation_records)
    matrix = transferability_matrix(rows, metric="mae")
    record = ExperimentRecord(
        experiment_id=stable_hash(
            {
                "dataset": dataset_id,
                "target": target_building,
                "model": canonical_model_name,
                "transfer": True,
            }
        ),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        building_id=target_building,
        train_period={
            "start": str(merged["timestamp"].min()),
            "end": str(merged["timestamp"].max()),
        },
        validation_period={},
        test_period={"target_building": target_building},
        model_name=f"transfer_{canonical_model_name}",
        model_architecture={
            "family": "pooled_leave_one_building_out",
            "zero_shot": True,
            "transfer_rows": rows,
            "transferability_matrix": matrix.to_dict() if not matrix.empty else {},
        },
        hyperparameters={"lookback_hours": lookback_hours, "horizon": horizon_name},
        feature_configuration=feature_configuration.value,
        input_window_steps=lookback_steps,
        forecast_horizon_steps=horizon_steps,
        random_seed=random_seed,
        corruption_configuration={},
        target_unit=str(merged.attrs.get("energy_target_unit", "Wh")),
        measurement_metadata={
            "energy_column": str(merged.attrs.get("energy_column", "energy_wh")),
            "energy_measurement_config": merged.attrs.get("energy_measurement_config", {}),
            "energy_conversion_summary": merged.attrs.get("energy_conversion_summary", {}),
            "occupancy_model": occupancy_model_name,
        },
        metrics={
            "zero_shot_mae": next(
                (float(row["mae"]) for row in rows if row["setting"] == "zero_shot"),
                None,
            ),
        },
        status="completed",
        data_is_demo=False,
    )
    manager = ExperimentManager(experiment_root)
    output = record.to_dict()
    output["transfer_rows"] = rows
    output["transferability_matrix"] = matrix.to_dict() if not matrix.empty else {}
    manager.save(record)
    return output
