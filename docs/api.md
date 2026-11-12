# API outline

The FastAPI application exposes OpenAPI documentation at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Service and database health |
| GET | `/api/datasets` | Registered datasets and audit status |
| POST | `/api/datasets/audit` | Audit an approved local dataset directory |
| GET | `/api/buildings` | Building metadata discovered from registered data |
| GET | `/api/buildings/{building_id}/summary` | Coverage and quality summary |
| POST | `/api/occupancy/train` | Start an occupancy training job |
| POST | `/api/occupancy/predict` | Generate occupancy predictions |
| POST | `/api/energy/train` | Start an energy training job |
| POST | `/api/energy/predict` | Generate a leakage-safe energy forecast |
| POST | `/api/experiments/run` | Start a tracked experiment |
| GET | `/api/experiments` | List persisted experiments |
| GET | `/api/experiments/{experiment_id}` | Inspect experiment metadata/results |
| GET | `/api/experiments/{experiment_id}/forecasts` | Read the persisted compact forecast preview |
| POST | `/api/experiments/compare` | Compare saved metric rows |
| POST | `/api/experiments/robustness` | Start a corruption experiment |
| POST | `/api/experiments/transfer` | Start leave-one-building-out/few-shot evaluation |
| GET | `/api/explanations/{experiment_id}` | Retrieve saved explanations |
| POST | `/api/scenarios` | Run a model-based, non-causal what-if scenario |
| POST | `/api/reports/generate` | Generate an HTML research report from actual results |
| GET | `/api/reports/{experiment_id}.html` | Download a generated HTML report |

Training and experiment endpoints return a job identifier and run in background tasks. Long-running production workloads should be moved to an isolated worker/queue without changing the request schemas.
