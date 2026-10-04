from __future__ import annotations

import threading
import time

import pytest

from nlp_tda import jobs


def _wait(job: jobs.Job) -> jobs.Job:
    for _ in range(200):
        if job.status != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


@pytest.fixture(autouse=True)
def _no_jobs():
    jobs._jobs.clear()
    yield
    jobs._jobs.clear()


def test_a_job_reports_progress_and_then_the_result(monkeypatch):
    release = threading.Event()

    def run_pipeline(source, *, progress):
        progress("extracting", 1, 3)
        release.wait(2)
        return {"run_id": "r1", "entities": 4}

    monkeypatch.setattr(jobs, "run_pipeline", run_pipeline)
    job = jobs.start(None)
    for _ in range(200):
        if job.stage == "extracting":
            break
        time.sleep(0.01)
    assert jobs.as_dict(jobs.get(job.job_id)) == {
        "job_id": job.job_id, "status": "running", "stage": "extracting", "done": 1, "total": 3,
        "result": None, "error": None,
    }
    with pytest.raises(jobs.JobRunning) as second:                # one run at a time
        jobs.start(None)
    assert second.value.job_id == job.job_id
    release.set()
    assert _wait(job).status == "completed" and job.result == {"run_id": "r1", "entities": 4}
    assert jobs.start(None).job_id != job.job_id                  # a new run can start afterwards


def test_a_failed_run_keeps_its_error(monkeypatch):
    def run_pipeline(source, *, progress):
        raise RuntimeError("The local LLM at http://ollama:11434 is not reachable.")

    monkeypatch.setattr(jobs, "run_pipeline", run_pipeline)
    job = _wait(jobs.start(None))
    assert job.status == "failed" and "not reachable" in job.error and job.result is None
