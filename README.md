# Campus Energy Forecasting

Privacy-preserving occupancy-aware probabilistic multi-horizon energy forecasting with cross-building generalization.

This repository implements a research platform built around the DECODE methodology and extends it with:

- forecasted-occupancy integration rather than operational use of future occupancy observations;
- direct 10-minute, 1-hour, 6-hour and 24-hour multi-horizon evaluation;
- chronological and leave-one-building-out evaluation;
- quantile forecasts with validation-only conformal calibration;
- controlled missing-data and sensor-outage experiments;
- tree-model explainability and model-based scenario analysis; and
- an API, experiment metadata store and research dashboard foundation.

Supported development runtimes: Python 3.11–3.13 and Node.js 20.19+ (Node 22 LTS or newer recommended). Frontend dependencies are pinned in `frontend/package.json` and `frontend/package-lock.json`.

The repository starts without I-BLEND files. The ingestion layer is ready for the published CSV files and refuses to describe demo data as research data. Synthetic fixtures, when used by tests or the local demo, are marked `DEMO DATA` and are excluded from research reports.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .[dev]
uvicorn campus_forecast.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the OpenAPI interface. The API is intentionally useful before a dataset is supplied: it exposes the audit and metadata endpoints, while training endpoints return a clear data-availability or unit-semantics error until valid files and explicit energy measurement configuration are present.

Run both containers with `docker compose up --build`; the API is available at `http://127.0.0.1:8000` and the dashboard at `http://127.0.0.1:5173`.

For a local frontend development server:

```bash
cd frontend
npm ci
npm run dev
npm test
```

The production frontend check is `npm run build`. Backend checks are `python -m pytest -q` and `python -m ruff check src tests`.

For optional PyTorch, LightGBM and SHAP support:

```bash
python -m pip install -e .[dev,research]
```

## Dataset preparation

Place the licensed I-BLEND files under `data/raw/i_blend/` or supply explicit paths to the CLI. See [`docs/dataset-preparation.md`](docs/dataset-preparation.md) for accepted schemas, explicit W/kW/Wh/kWh configuration, timestamp handling and the weather-coverage limitation documented by the dataset authors.

Audit a directory without training a model:

```bash
python -m campus_forecast.cli audit --root data/raw/i_blend --output artifacts/reports/data-availability.json
```

This audit intentionally remains unresolved until the source semantics are declared. After verifying the downloaded file documentation, pass the matching configuration, for example:

```bash
campus-audit --root data/raw/i_blend --output artifacts/reports/i-blend-audit.json \
  --source-unit W --semantics instantaneous_power --target-unit Wh \
  --expected-interval-minutes 1
```

Run a small software-only demonstration:

```bash
python -m campus_forecast.cli demo --output artifacts/reports/demo-results.json
```

The demo is not an I-BLEND result and cannot be used by research-report generation.

## Research integrity

All forecasting splits are chronological. Scalers and imputers are fit on training observations only. Operational occupancy-aware forecasting consumes occupancy forecasts; actual future occupancy is available only through an explicitly named oracle diagnostic. Observed future weather is never used as an operational feature. Validation/calibration periods remain separate from final test periods.

Reported DECODE values are reference values from the cited paper, not results produced by this repository. The implementation records assumptions where the paper does not fully specify window construction, aggregation, or the original code path.

## Project map

| Area | Location |
| --- | --- |
| Data ingestion and audit | `src/campus_forecast/data/` |
| Leakage-safe preprocessing and windows | `src/campus_forecast/pipeline/` |
| Models and forecasting configurations | `src/campus_forecast/models/` |
| Metrics and experiments | `src/campus_forecast/evaluation/` |
| API and persistence | `src/campus_forecast/api/`, `src/campus_forecast/storage.py` |
| Frontend | `frontend/` |
| Tests | `tests/` |
| Methodology and limitations | `docs/` |

## Citations

Mishra, A., Lone, H. R., & Mishra, A. (2024). *DECODE: Data-driven Energy Consumption Prediction leveraging Historical Data and Environmental Factors in Buildings*. Energy and Buildings, 113950. DOI: [10.1016/j.enbuild.2024.113950](https://doi.org/10.1016/j.enbuild.2024.113950). Preprint: [arXiv:2309.02908](https://arxiv.org/abs/2309.02908).

Rashid, H., Singh, P., & Singh, A. (2019). *I-BLEND, a campus scale commercial and residential buildings electrical energy dataset*. Scientific Data, 6, 190015. DOI: [10.1038/sdata.2019.15](https://doi.org/10.1038/sdata.2019.15).
