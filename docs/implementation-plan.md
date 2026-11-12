# Implementation plan and current status

Updated: 2026-11-03.

## Stage 1 — research and repository audit

Status: implemented.

- The workspace was empty at project start.
- The DECODE arXiv HTML was reviewed. Reproducible details include a 70:15:15 chronological split, min-max normalization, 10-minute common resolution, occupancy/temperature/humidity/calendar features, and a one-layer 32-unit LSTM followed by two five-unit dense layers.
- The I-BLEND data descriptor was reviewed. Energy is released at one-minute resolution, occupancy at ten-minute resolution, weather is supplementary and limited, and calendar data is date-based.
- Ambiguities are recorded in `docs/methodology.md`.

## Stage 2 — ingestion and validation

Status: implemented and unit-audit gated.

- CSV/Parquet/JSON table loading with file-size and extension checks.
- Timestamp normalization to `Asia/Kolkata`.
- Alias-aware canonical schema mapping with original source columns preserved.
- Explicit W/kW/Wh/kWh and instantaneous/interval/cumulative semantics.
- Actual-interval power integration, irregular-gap policy and target-unit metadata.
- Duplicate, gap, unit, range and missingness audits.
- Building-wise profiles and weather coverage reporting.
- No silent synthetic weather substitution.

## Stage 3 — DECODE baseline

Status: implemented and service-integrated; real-data benchmark pending dataset files and explicit unit configuration.

- Persistence and seasonal-naive baselines.
- Linear regression, decision tree and random forest direct multi-horizon baselines.
- Optional PyTorch DECODE-style LSTM with validation checkpoint selection, direct multi-horizon output and persisted architecture metadata.

## Stage 4 — occupancy integration

Status: implemented, issue-time leakage-tested and service-integrated; full benchmark pending data.

- Seasonal-naive occupancy and HistGradientBoosting occupancy forecaster.
- Issue-time-keyed out-of-sample occupancy predictions; future occupancy observations are never used for the predicted configuration.
- Energy configurations for energy-only, predicted-occupancy and oracle diagnostics.

## Stages 5–7 — transfer, probabilistic forecasting, robustness and explainability

Status: core reusable modules implemented; execution requires real data and is recorded through the experiment manager.

- Leave-one-building-out and few-shot split planning.
- Quantile forecasting and validation-only conformal calibration.
- Random/consecutive missingness, noise and causal imputation strategies.
- Permutation/tree/optional SHAP explainability and what-if scenarios.

## Stages 8–10 — dashboard, benchmark execution and reports

Status: dashboard/API workflows are connected for audit, occupancy, energy, transfer, robustness, scenarios and reports. Full research reports are intentionally not generated until the dataset audit and experiments have real inputs.

Next execution step after supplying I-BLEND is:

1. run and inspect the data availability report;
2. select valid common periods per building and weather configuration;
3. run the clean baseline suite;
4. run occupancy, transfer, uncertainty and robustness experiments;
5. generate reports containing actual persisted results and limitations.

## Verification status

- **Unit-tested:** explicit W/kW/Wh/kWh conversion, actual-duration integration, irregular-gap handling, issue-time occupancy leakage, chronological occupancy evaluation, transfer folds and robustness corruption.
- **Integration-tested:** service training/probabilistic/occupancy/tree/LSTM wiring, persistence metadata, API validation and absent-dataset failure paths.
- **Frontend-tested:** audit and energy-job interactions with mocked API responses; clean `npm ci`, `npm test` and production build.
- **Synthetic fixtures only:** service and model integration metrics in the test suite. They are not research results.
- **Real I-BLEND:** not evaluated in this workspace because licensed I-BLEND source files are absent. No I-BLEND benchmark metrics or improvements are claimed.
