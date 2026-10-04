"""Pipeline runs as background jobs: start one, then poll it for progress and the result.

One run at a time (they share the local LLM). Jobs live in memory; a restart forgets them.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nlp_tda.pipeline import run_pipeline


class JobRunning(RuntimeError):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"A pipeline run is already in progress (job {job_id}).")
        self.job_id = job_id


@dataclass
class Job:
    job_id: str
    status: str = "running"  # running | completed | failed
    stage: str = "starting"  # parsing | embedding | themes | extracting | saving
    done: int = 0  # steps finished in this stage (LLM calls while extracting)
    total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def start(source: Path | None = None) -> Job:
    """Start a pipeline run on ``source`` in the background. Raises ``JobRunning`` if one is active."""
    with _lock:
        active = next((j for j in _jobs.values() if j.status == "running"), None)
        if active:
            raise JobRunning(active.job_id)
        job = Job(job_id=uuid.uuid4().hex[:12])
        _jobs[job.job_id] = job

    def progress(stage: str, done: int = 0, total: int = 0) -> None:
        job.stage, job.done, job.total = stage, done, total

    def work() -> None:
        try:
            job.result = run_pipeline(source, progress=progress)
            job.status = "completed"
        except Exception as exc:
            job.error = str(exc) or type(exc).__name__
            job.status = "failed"

    threading.Thread(target=work, name=f"pipeline-{job.job_id}", daemon=True).start()
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def as_dict(job: Job) -> dict[str, Any]:
    return asdict(job)
