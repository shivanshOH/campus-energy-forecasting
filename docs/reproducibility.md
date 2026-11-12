# Reproducibility workflow

## Environment

Record the Python version, package lock/install command, operating system, CPU/GPU availability and random seed in the experiment metadata. The default implementation is CPU compatible. PyTorch, LightGBM and SHAP are optional extras and are recorded as unavailable when not installed.

Supported development versions are Python 3.11–3.13 and Node.js 20.19 or newer. Install the frontend with `npm ci`; the committed `frontend/package-lock.json` is the reproducibility lockfile.

## Real-data workflow

1. Obtain I-BLEND from its official dataset distribution and preserve its license/citation.
2. Place the source files under `data/raw/i_blend/` using the layout in `dataset-preparation.md`.
3. Run `campus-audit` and retain the generated JSON data-availability report. Supply explicit measurement semantics through the API audit request before treating the audit as research-ready.
4. Restrict every experiment to the observed building/modality overlap reported by the audit.
5. Run clean baselines before occupancy, transfer, probabilistic or corruption experiments.
6. Keep validation/calibration and final test periods separate.
7. Persist experiment metadata, metrics, checkpoints and source coverage together.
8. Generate an HTML report only from a persisted non-demo experiment.

## Demo-data boundary

`campus-demo` creates a synthetic software fixture for smoke testing. It is marked `DEMO DATA`, written to a separate report path, and rejected by research-report generation. It is not a benchmark and must not be cited as I-BLEND performance.

## Leakage checks

- Source timestamps are sorted and aligned before windowing.
- Window targets are strictly after the issue timestamp.
- Train-only causal imputers and scalers are fit before validation/test transformation.
- Future occupancy enters an operational energy configuration only through `occupancy_pred`.
- Future observed weather never enters an issue table; only historical weather values are used.
- Oracle occupancy is a separate diagnostic configuration and requires an explicit diagnostic flag at the API boundary.
- Few-shot adaptation rows are separated from subsequent evaluation rows.
