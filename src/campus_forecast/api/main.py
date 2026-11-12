"""FastAPI entry point for the campus forecasting platform."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi import Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import text

from campus_forecast.api.jobs import JobStore
from campus_forecast.api.schemas import (
    AuditRequest,
    CompareRequest,
    ExperimentRunRequest,
    ForecastRequest,
    JobResponse,
    JobStatusResponse,
    ReportRequest,
    RobustnessRequest,
    ScenarioRunRequest,
    TrainRequest,
    TransferRequest,
)
from campus_forecast.config import get_settings
from campus_forecast.data.audit import audit_directory
from campus_forecast.evaluation.explainability import (
    ScenarioRequest as DomainScenarioRequest,
)
from campus_forecast.evaluation.explainability import (
    scenario_prediction,
    tree_feature_importance,
)
from campus_forecast.evaluation.robustness import CorruptionConfig
from campus_forecast.exceptions import CampusForecastError
from campus_forecast.reporting import generate_html_report
from campus_forecast.schema import ExperimentRecord, FeatureConfiguration, ImputationStrategy
from campus_forecast.services import (
    run_occupancy_experiment,
    run_probabilistic_experiment,
    run_robustness_experiment,
    run_training_experiment,
    run_transfer_experiment,
)
from campus_forecast.storage import (
    DatasetRow,
    ExperimentRow,
    ForecastRow,
    db_session,
    init_db,
    list_experiment_rows,
    save_experiment,
    save_forecast_preview,
    upsert_dataset,
)
from campus_forecast.utils import json_dump

settings = get_settings()
settings.ensure_directories()
init_db()
jobs = JobStore()

app = FastAPI(
    title="Campus Energy Forecasting Platform",
    description=(
        "Research API for leakage-safe occupancy-aware probabilistic multi-horizon "
        "building energy forecasting."
    ),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _safe_path(raw_path: str) -> Path:
    root = settings.data_root.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Dataset path must be inside the configured data root"
        ) from exc
    return candidate


def _dataset_or_404(session, dataset_id: str) -> DatasetRow:
    dataset = session.get(DatasetRow, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset not registered: {dataset_id}")
    return dataset


def _job_response(job, message: str) -> JobResponse:
    return JobResponse(job_id=job.job_id, status=job.status, message=message)


def _energy_config(request: Any):
    measurement = getattr(request, "energy_measurement", None)
    return measurement.to_domain() if measurement is not None else None


def _require_energy_measurement(request: Any) -> None:
    if getattr(request, "energy_measurement", None) is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "EnergyMeasurementConfig is required for energy experiments; "
                "declare source unit, semantics, target unit and interval policy explicitly."
            ),
        )


def _persist_result(result: dict[str, Any]) -> None:
    field_names = {field.name for field in fields(ExperimentRecord)}
    record_data = {
        key: value for key, value in result.items() if key in field_names and key != "created_at"
    }
    record = ExperimentRecord(**record_data)
    with db_session() as session:
        save_experiment(session, record)
        save_forecast_preview(session, result)


def _run_and_persist(task) -> dict[str, Any]:
    result = task()
    _persist_result(result)
    return result


@app.get("/api/health")
def health() -> dict[str, str]:
    session = db_session()
    try:
        session.execute(text("SELECT 1"))
    finally:
        session.close()
    return {"status": "ok", "environment": settings.app_env, "version": app.version}


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    try:
        return JobStatusResponse(**jobs.as_dict(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, Any]]:
    session = db_session()
    try:
        rows = session.query(DatasetRow).order_by(DatasetRow.created_at.desc()).all()
        return [
            {
                "dataset_id": row.dataset_id,
                "identifier": row.identifier,
                "version": row.version,
                "root_path": row.root_path,
                "audit": row.audit,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    finally:
        session.close()


@app.post("/api/datasets/audit")
def audit_dataset(request: AuditRequest) -> dict[str, Any]:
    root = _safe_path(request.root_path)
    try:
        result = audit_directory(
            root,
            max_file_bytes=settings.max_upload_bytes,
            energy_config=_energy_config(request),
        )
    except (FileNotFoundError, CampusForecastError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_path = settings.reports_dir / f"{request.dataset_id}-data-availability.json"
    json_dump(result, audit_path)
    session = db_session()
    try:
        upsert_dataset(
            session,
            dataset_id=request.dataset_id,
            identifier="I-BLEND" if request.dataset_id == "i_blend" else request.dataset_id,
            version=request.dataset_version,
            root_path=str(root),
            audit=result,
        )
    finally:
        session.close()
    return {"dataset_id": request.dataset_id, "audit_path": str(audit_path), **result}


@app.get("/api/buildings")
def list_buildings() -> list[dict[str, Any]]:
    session = db_session()
    try:
        result: dict[str, dict[str, Any]] = {}
        for dataset in session.query(DatasetRow).all():
            coverage = (dataset.audit or {}).get("coverage_by_kind", {})
            for building in coverage.get("energy", {}).get("buildings", []):
                result.setdefault(
                    building,
                    {
                        "building_id": building,
                        "name": building,
                        "category": "unknown",
                        "datasets": [],
                    },
                )
                result[building]["datasets"].append(dataset.dataset_id)
        return sorted(result.values(), key=lambda value: value["building_id"])
    finally:
        session.close()


@app.get("/api/buildings/{building_id}/summary")
def building_summary(building_id: str = ApiPath(min_length=1, max_length=128)) -> dict[str, Any]:
    session = db_session()
    try:
        summaries = []
        for dataset in session.query(DatasetRow).all():
            audit = dataset.audit or {}
            matching = [
                report
                for report in audit.get("reports", [])
                if building_id in report.get("buildings", [])
            ]
            if matching:
                summaries.append({"dataset_id": dataset.dataset_id, "reports": matching})
        if not summaries:
            raise HTTPException(status_code=404, detail=f"Building not found: {building_id}")
        return {"building_id": building_id, "summaries": summaries}
    finally:
        session.close()


def _training_task(request: TrainRequest, dataset: DatasetRow) -> dict[str, Any]:
    if request.model_name.lower() in {"quantile", "quantile_direct", "lightgbm_quantile"}:
        return run_probabilistic_experiment(
            root=dataset.root_path,
            dataset_id=dataset.dataset_id,
            dataset_version="registered",
            building_id=request.building_id,
            horizon_name=request.horizon,
            lookback_hours=request.lookback_hours,
            feature_configuration=request.feature_configuration,
            random_seed=request.random_seed,
            energy_config=_energy_config(request),
            occupancy_model_name=request.occupancy_model_name,
        )
    return run_training_experiment(
        root=dataset.root_path,
        dataset_id=dataset.dataset_id,
        dataset_version="registered",
        building_id=request.building_id,
        model_name=request.model_name,
        horizon_name=request.horizon,
        lookback_hours=request.lookback_hours,
        feature_configuration=request.feature_configuration,
        random_seed=request.random_seed,
        energy_config=_energy_config(request),
        occupancy_model_name=request.occupancy_model_name,
    )


def _submit_training(background_tasks: BackgroundTasks, request: TrainRequest, kind: str):
    session = db_session()
    try:
        dataset = _dataset_or_404(session, request.dataset_id)
        job = jobs.submit(
            background_tasks,
            lambda: _run_and_persist(lambda: _training_task(request, dataset)),
        )
    finally:
        session.close()
    return _job_response(job, f"{kind} training queued; poll /api/jobs/{job.job_id}")


def _submit_occupancy(background_tasks: BackgroundTasks, request: TrainRequest, kind: str):
    session = db_session()
    try:
        dataset = _dataset_or_404(session, request.dataset_id)
        job = jobs.submit(
            background_tasks,
            lambda: _run_and_persist(
                lambda: run_occupancy_experiment(
                    root=dataset.root_path,
                    dataset_id=dataset.dataset_id,
                    dataset_version="registered",
                    building_id=request.building_id,
                    horizon_name=request.horizon,
                    random_seed=request.random_seed,
                    model_name=request.occupancy_model_name,
                    lookback_hours=request.lookback_hours,
                )
            ),
        )
    finally:
        session.close()
    return _job_response(job, f"{kind} queued; poll /api/jobs/{job.job_id}")


@app.post("/api/occupancy/train", response_model=JobResponse)
def train_occupancy(request: TrainRequest, background_tasks: BackgroundTasks) -> JobResponse:
    if request.feature_configuration.value.startswith(("A_", "B_", "C_")):
        raise HTTPException(
            status_code=422,
            detail="Occupancy training requires an occupancy-aware feature configuration",
        )
    if request.feature_configuration.value == "F_oracle_occupancy_diagnostic":
        raise HTTPException(status_code=422, detail="Oracle occupancy is an energy diagnostic only")
    return _submit_occupancy(background_tasks, request, "Occupancy")


@app.post("/api/occupancy/predict", response_model=JobResponse)
def predict_occupancy(request: ForecastRequest, background_tasks: BackgroundTasks) -> JobResponse:
    if request.occupancy_mode.value == "oracle":
        raise HTTPException(status_code=422, detail="Oracle occupancy is an energy diagnostic only")
    train_request = TrainRequest(
        dataset_id=request.dataset_id,
        building_id=request.building_id,
        model_name=request.model_name,
        horizon=request.horizon,
        feature_configuration=request.feature_configuration,
        occupancy_model_name=request.occupancy_model_name,
    )
    return _submit_occupancy(background_tasks, train_request, "Occupancy prediction")


@app.post("/api/energy/train", response_model=JobResponse)
def train_energy(request: TrainRequest, background_tasks: BackgroundTasks) -> JobResponse:
    _require_energy_measurement(request)
    if request.feature_configuration.value == "F_oracle_occupancy_diagnostic":
        raise HTTPException(
            status_code=422, detail="Oracle occupancy requires an explicit diagnostic request"
        )
    return _submit_training(background_tasks, request, "Energy")


@app.post("/api/energy/predict", response_model=JobResponse)
def predict_energy(request: ForecastRequest, background_tasks: BackgroundTasks) -> JobResponse:
    _require_energy_measurement(request)
    if request.occupancy_mode.value == "oracle" and not request.diagnostic:
        raise HTTPException(status_code=422, detail="Oracle occupancy is diagnostic-only")
    train_request = TrainRequest(
        dataset_id=request.dataset_id,
        building_id=request.building_id,
        model_name=request.model_name,
        horizon=request.horizon,
        feature_configuration=(
            request.feature_configuration
            if request.occupancy_mode.value != "oracle"
            else FeatureConfiguration.ORACLE_OCCUPANCY
        ),
        occupancy_model_name=request.occupancy_model_name,
        energy_measurement=request.energy_measurement,
    )
    return _submit_training(background_tasks, train_request, "Energy prediction")


@app.post("/api/experiments/run", response_model=JobResponse)
def run_experiment(request: ExperimentRunRequest, background_tasks: BackgroundTasks) -> JobResponse:
    _require_energy_measurement(request)
    session = db_session()
    try:
        dataset = _dataset_or_404(session, request.dataset_id)

        def task() -> dict[str, Any]:
            if request.model_name.lower() in {"quantile", "quantile_direct", "lightgbm_quantile"}:
                result = run_probabilistic_experiment(
                    root=dataset.root_path,
                    dataset_id=dataset.dataset_id,
                    dataset_version=request.dataset_version,
                    building_id=request.building_id,
                    horizon_name=request.horizon,
                    lookback_hours=request.lookback_hours,
                    feature_configuration=request.feature_configuration,
                    random_seed=request.random_seed,
                    energy_config=_energy_config(request),
                    occupancy_model_name=request.occupancy_model_name,
                )
            else:
                result = run_training_experiment(
                    root=dataset.root_path,
                    dataset_id=dataset.dataset_id,
                    dataset_version=request.dataset_version,
                    building_id=request.building_id,
                    model_name=request.model_name,
                    horizon_name=request.horizon,
                    lookback_hours=request.lookback_hours,
                    feature_configuration=request.feature_configuration,
                    random_seed=request.random_seed,
                    energy_config=_energy_config(request),
                    occupancy_model_name=request.occupancy_model_name,
                )
            # The JSON artifact remains the source of full metadata; the
            # relational row provides queryable experiment summaries.
            _persist_result(result)
            return result

        job = jobs.submit(background_tasks, task)
    finally:
        session.close()
    return _job_response(job, f"Experiment queued; poll /api/jobs/{job.job_id}")


@app.post("/api/experiments/probabilistic", response_model=JobResponse)
def run_probabilistic(
    request: ExperimentRunRequest, background_tasks: BackgroundTasks
) -> JobResponse:
    _require_energy_measurement(request)
    request = request.model_copy(update={"model_name": "quantile_direct"})
    return run_experiment(request, background_tasks)


@app.get("/api/experiments")
def list_experiments() -> list[dict[str, Any]]:
    session = db_session()
    try:
        return [
            {
                "experiment_id": row.experiment_id,
                "dataset_id": row.dataset_id,
                "model_name": row.model_name,
                "status": row.status,
                "building_id": row.building_id,
                "feature_configuration": row.feature_configuration,
                "metrics": row.metrics,
                "metadata": row.metadata_json,
            }
            for row in list_experiment_rows(session)
        ]
    finally:
        session.close()


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    session = db_session()
    try:
        row = session.get(ExperimentRow, experiment_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return {
            "experiment_id": row.experiment_id,
            "dataset_id": row.dataset_id,
            "model_name": row.model_name,
            "status": row.status,
            "building_id": row.building_id,
            "feature_configuration": row.feature_configuration,
            "metrics": row.metrics,
            "metadata": row.metadata_json,
        }
    finally:
        session.close()


@app.get("/api/experiments/{experiment_id}/forecasts")
def get_forecasts(experiment_id: str) -> list[dict[str, Any]]:
    session = db_session()
    try:
        rows = (
            session.query(ForecastRow)
            .filter(ForecastRow.experiment_id == experiment_id)
            .order_by(ForecastRow.issue_timestamp, ForecastRow.horizon_step)
            .all()
        )
        return [
            {
                "experiment_id": row.experiment_id,
                "building_id": row.building_id,
                "issue_timestamp": row.issue_timestamp.isoformat(),
                "horizon_step": row.horizon_step,
                "p10": row.p10,
                "p50": row.p50,
                "p90": row.p90,
                "actual": row.actual,
            }
            for row in rows
        ]
    finally:
        session.close()


@app.post("/api/experiments/compare")
def compare_experiments(request: CompareRequest) -> dict[str, Any]:
    session = db_session()
    try:
        rows = [
            session.get(ExperimentRow, experiment_id) for experiment_id in request.experiment_ids
        ]
        missing = [
            experiment_id
            for experiment_id, row in zip(request.experiment_ids, rows, strict=True)
            if row is None
        ]
        if missing:
            raise HTTPException(status_code=404, detail=f"Experiments not found: {missing}")
        return {
            "experiments": [
                {
                    "experiment_id": row.experiment_id,
                    "model_name": row.model_name,
                    "metrics": row.metrics,
                }
                for row in rows
            ]
        }
    finally:
        session.close()


@app.post("/api/experiments/robustness", response_model=JobResponse)
def run_robustness(request: RobustnessRequest, background_tasks: BackgroundTasks) -> JobResponse:
    _require_energy_measurement(request)
    # The shared service is intentionally queued; the actual metric run uses
    # the same registered data path and corruption configuration as the CLI.
    session = db_session()
    try:
        dataset = _dataset_or_404(session, request.dataset_id)
        config = CorruptionConfig(
            seed=request.random_seed,
            occupancy_missing_percent=request.occupancy_missing_percent,
            weather_missing_percent=request.weather_missing_percent,
            outage_hours=request.outage_hours,
            occupancy_noise_std=request.occupancy_noise_std,
        )
        try:
            strategy = ImputationStrategy(request.imputation_strategy)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Unknown imputation strategy") from exc
        job = jobs.submit(
            background_tasks,
            lambda: _run_and_persist(
                lambda: run_robustness_experiment(
                    root=dataset.root_path,
                    dataset_id=dataset.dataset_id,
                    dataset_version="registered",
                    building_id=request.building_id,
                    model_name=request.model_name,
                    horizon_name=request.horizon,
                    lookback_hours=request.lookback_hours,
                    feature_configuration=request.feature_configuration,
                    corruption=config,
                    imputation_strategy=strategy,
                    random_seed=request.random_seed,
                    energy_config=_energy_config(request),
                    occupancy_model_name=request.occupancy_model_name,
                )
            ),
        )
    finally:
        session.close()
    return _job_response(job, f"Robustness experiment queued; poll /api/jobs/{job.job_id}")


@app.post("/api/experiments/transfer", response_model=JobResponse)
def run_transfer(request: TransferRequest, background_tasks: BackgroundTasks) -> JobResponse:
    _require_energy_measurement(request)
    session = db_session()
    try:
        dataset = _dataset_or_404(session, request.dataset_id)

        def task() -> dict[str, Any]:
            return _run_and_persist(
                lambda: run_transfer_experiment(
                    root=dataset.root_path,
                    dataset_id=dataset.dataset_id,
                    dataset_version="registered",
                    target_building=request.target_building,
                    model_name=request.model_name,
                    horizon_name=request.horizon,
                    lookback_hours=request.lookback_hours,
                    feature_configuration=request.feature_configuration,
                    adaptation_days=tuple(request.adaptation_days),
                    random_seed=request.random_seed,
                    energy_config=_energy_config(request),
                    occupancy_model_name=request.occupancy_model_name,
                )
            )

        job = jobs.submit(background_tasks, task)
    finally:
        session.close()
    return _job_response(job, f"Transfer experiment queued; poll /api/jobs/{job.job_id}")


@app.get("/api/explanations/{experiment_id}")
def get_explanation(experiment_id: str) -> dict[str, Any]:
    experiment = get_experiment(experiment_id)
    metadata = experiment.get("metadata") or {}
    checkpoint_value = metadata.get("checkpoint_path")
    feature_importance: list[dict[str, Any]] = []
    note = "No fitted model artifact is available for this experiment."
    if checkpoint_value:
        checkpoint = Path(str(checkpoint_value)).resolve()
        try:
            checkpoint.relative_to(settings.checkpoints_dir.resolve())
            import joblib

            payload = joblib.load(checkpoint)
            model = payload.get("model")
            estimators = getattr(model, "estimators", None)
            if estimators:
                feature_importance = tree_feature_importance(
                    estimators[0], payload.get("feature_columns", [])
                ).to_dict(orient="records")
                note = "Tree feature importance from the first direct horizon estimator."
        except Exception as exc:
            note = f"Saved artifact could not be explained: {exc}"
    return {
        "experiment_id": experiment_id,
        "feature_importance": feature_importance,
        "shap": None,
        "note": note,
        "experiment": experiment,
    }


@app.post("/api/scenarios")
def run_scenario(request: ScenarioRunRequest) -> dict[str, Any]:
    experiment = get_experiment(request.experiment_id)
    checkpoint_value = (experiment.get("metadata") or {}).get("checkpoint_path")
    if not checkpoint_value:
        raise HTTPException(
            status_code=422,
            detail="This experiment has no saved model checkpoint for scenario analysis",
        )
    checkpoint = Path(str(checkpoint_value)).resolve()
    try:
        checkpoint.relative_to(settings.checkpoints_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unsafe checkpoint path") from exc
    if not checkpoint.exists():
        raise HTTPException(status_code=404, detail="Saved model checkpoint not found")
    try:
        import joblib

        payload = joblib.load(checkpoint)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not load model checkpoint: {exc}"
        ) from exc
    reference = payload.get("reference_features")
    model = payload.get("model")
    if reference is None or model is None:
        raise HTTPException(status_code=422, detail="Checkpoint has no scenario reference features")
    return scenario_prediction(
        model,
        reference,
        DomainScenarioRequest(
            occupancy_multiplier=request.occupancy_multiplier,
            temperature_delta_c=request.temperature_delta_c,
            humidity_delta_pct=request.humidity_delta_pct,
            working_day_override=request.working_day_override,
        ),
    )


@app.post("/api/reports/generate")
def generate_report(request: ReportRequest) -> dict[str, str]:
    experiment = get_experiment(request.experiment_id)
    metadata = experiment.get("metadata", {})
    if metadata.get("data_is_demo", False):
        raise HTTPException(
            status_code=422, detail="DEMO DATA cannot be included in a research report"
        )
    output = settings.reports_dir / f"{request.experiment_id}.html"
    path = generate_html_report({**metadata, "metrics": experiment.get("metrics", {})}, output)
    return {"experiment_id": request.experiment_id, "path": str(path)}


@app.get("/api/reports/{experiment_id}.html")
def download_report(experiment_id: str) -> FileResponse:
    path = (settings.reports_dir / f"{experiment_id}.html").resolve()
    try:
        path.relative_to(settings.reports_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unsafe report path") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report has not been generated")
    return FileResponse(path, media_type="text/html", filename=path.name)


def run() -> None:
    import uvicorn

    uvicorn.run("campus_forecast.api.main:app", host="127.0.0.1", port=8000, reload=False)
