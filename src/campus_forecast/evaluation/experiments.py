"""File-backed experiment metadata manager.

Large arrays and checkpoints remain files; the manager stores a compact,
portable JSON record suitable for later insertion into SQLite/PostgreSQL.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from campus_forecast.evaluation.metrics import evaluate_point
from campus_forecast.schema import ExperimentRecord
from campus_forecast.utils import json_dump


class ExperimentManager:
    def __init__(self, root: str | Path = "artifacts/experiments") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_record(self, **kwargs: Any) -> ExperimentRecord:
        kwargs.setdefault("experiment_id", uuid.uuid4().hex)
        kwargs.setdefault("status", "created")
        return ExperimentRecord(**kwargs)

    def save(self, record: ExperimentRecord) -> Path:
        return json_dump(record.to_dict(), self.root / f"{record.experiment_id}.json")

    def load(self, experiment_id: str) -> dict[str, Any]:
        path = self.root / f"{experiment_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.root.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return records

    def run_point(
        self,
        record: ExperimentRecord,
        model: object,
        train_features: object,
        train_targets: object,
        test_features: object,
        test_targets: object,
    ) -> ExperimentRecord:
        started = time.perf_counter()
        record.status = "running"
        self.save(record)
        model.fit(np.asarray(train_features), np.asarray(train_targets))
        predictions = model.predict(np.asarray(test_features))
        actual = np.asarray(test_targets)
        if actual.ndim == 2:
            record.metrics = {
                f"horizon_{index + 1}_{name}": value
                for index in range(actual.shape[1])
                for name, value in evaluate_point(actual[:, index], predictions[:, index]).items()
            }
        else:
            record.metrics = evaluate_point(actual, predictions)
        record.training_duration_seconds = time.perf_counter() - started
        record.status = "completed"
        self.save(record)
        return record
