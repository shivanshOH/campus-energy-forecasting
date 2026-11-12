"""Feature attribution and explicitly non-causal scenario analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from campus_forecast.exceptions import OptionalDependencyError


def tree_feature_importance(model: object, feature_names: list[str]) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        raise ValueError("Model does not expose tree feature_importances_")
    values = np.asarray(values, dtype=float)
    if len(values) != len(feature_names):
        raise ValueError("feature_names length does not match model importance length")
    result = pd.DataFrame({"feature": feature_names, "importance": values})
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_feature_importance(
    model: object,
    features: object,
    targets: object,
    feature_names: list[str],
    *,
    scoring: str = "neg_mean_absolute_error",
    random_state: int = 42,
    n_repeats: int = 5,
) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        np.asarray(features),
        np.asarray(targets),
        scoring=scoring,
        random_state=random_state,
        n_repeats=n_repeats,
    )
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def shap_feature_importance(
    model: object, features: object, feature_names: list[str]
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise OptionalDependencyError("SHAP is required for SHAP explanations") from exc
    explainer = shap.Explainer(model, features)
    values = explainer(features).values
    values = np.asarray(values)
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1, values.shape[-1]).mean(axis=1)
    importance = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class ScenarioRequest:
    occupancy_multiplier: float = 1.0
    temperature_delta_c: float = 0.0
    humidity_delta_pct: float = 0.0
    working_day_override: float | None = None


def apply_scenario(features: pd.DataFrame, scenario: ScenarioRequest) -> pd.DataFrame:
    """Modify feature columns for a model-based what-if scenario."""

    if scenario.occupancy_multiplier < 0:
        raise ValueError("occupancy_multiplier must be non-negative")
    output = features.copy()
    occupancy_columns = [column for column in output if "occupancy" in str(column).lower()]
    temperature_columns = [column for column in output if "temperature" in str(column).lower()]
    humidity_columns = [column for column in output if "humidity" in str(column).lower()]
    output[occupancy_columns] = output[occupancy_columns] * scenario.occupancy_multiplier
    output[temperature_columns] = output[temperature_columns] + scenario.temperature_delta_c
    output[humidity_columns] = output[humidity_columns] + scenario.humidity_delta_pct
    if scenario.working_day_override is not None:
        calendar_columns = [column for column in output if "working_day" in str(column)]
        output[calendar_columns] = scenario.working_day_override
    return output


def scenario_prediction(
    model: object,
    features: pd.DataFrame,
    scenario: ScenarioRequest,
) -> dict[str, object]:
    baseline = np.asarray(model.predict(features), dtype=float)
    modified_features = apply_scenario(features, scenario)
    scenario_values = np.asarray(model.predict(modified_features), dtype=float)
    return {
        "label": "model_based_what_if",
        "interpretation": "Not a causal estimate and not a guarantee of energy savings.",
        "baseline": baseline.tolist(),
        "scenario": scenario_values.tolist(),
        "difference": (scenario_values - baseline).tolist(),
        "request": scenario.__dict__,
    }
