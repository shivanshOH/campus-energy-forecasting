"""In-process background job registry for local development.

The interface is intentionally queue-friendly: a production deployment can
replace the executor with Celery/RQ without changing API response contracts.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import BackgroundTasks


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    result: Any | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, background_tasks: BackgroundTasks, task: Callable[[], Any]) -> Job:
        job = Job(job_id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.job_id] = job
        background_tasks.add_task(self._run, job.job_id, task)
        return job

    def _run(self, job_id: str, task: Callable[[], Any]) -> None:
        job = self.get(job_id)
        with job.lock:
            job.status = "running"
        try:
            result = task()
            with job.lock:
                job.result = result
                job.status = "completed"
        except Exception as exc:  # API clients need a serializable failure
            with job.lock:
                job.error = str(exc)
                job.status = "failed"

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def as_dict(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        with job.lock:
            return {
                "job_id": job.job_id,
                "status": job.status,
                "result": job.result,
                "error": job.error,
            }
