from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nlp_tda import jobs
from nlp_tda.app import app
from nlp_tda.config import settings
from nlp_tda.db import reset_db_for_tests
from nlp_tda.export.excel_export import find_export
from nlp_tda.ingest.parsers import expand_zip
from nlp_tda.ingest.uploads import batch_dir, create_batch, save_upload


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch):
    for name, path in (("data_dir", "data"), ("sqlite_path", "data/master.db"), ("chroma_path", "data/chroma"),
                       ("uploads_dir", "data/uploads"), ("exports_dir", "data/exports")):
        monkeypatch.setattr(settings, name, tmp_path / path)
    monkeypatch.setattr(settings, "auth_password", "")
    reset_db_for_tests()
    yield
    reset_db_for_tests()


@pytest.fixture
def started(monkeypatch) -> list[Path]:
    """Capture what a run would be started on, without running anything."""
    sources: list[Path] = []

    def start(source):
        sources.append(Path(source))
        return jobs.Job(job_id="0123456789ab")

    monkeypatch.setattr(jobs, "start", start)
    return sources


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def test_a_caller_cannot_point_the_pipeline_at_a_server_folder(started):
    client = TestClient(app)
    assert client.post("/api/pipeline/run", params={"source_dir": "/etc"}).status_code == 200
    assert started == [Path(settings.fixtures_dir)]                    # the parameter no longer exists
    for bad in ("../..", "..%2F..", "/etc", "0123456789ab", "*"):      # not a batch id, or no such batch
        assert client.post("/api/pipeline/run", params={"batch_id": bad}).status_code == 400
    assert len(started) == 1
    batch = create_batch()
    assert client.post("/api/pipeline/run", params={"batch_id": batch.name}).status_code == 200
    assert started[-1] == batch


def test_batch_and_export_ids_only_name_their_own_files():
    with pytest.raises(ValueError):
        batch_dir("../../etc")
    assert find_export("*") is None and find_export("../x") is None


def test_zip_members_cannot_leave_the_target_folder(tmp_path):
    archive = tmp_path / "pack.zip"
    archive.write_bytes(_zip({"../../evil.txt": b"x", "/abs/notes.txt": b"y"}))
    extracted = expand_zip(archive, tmp_path / "out")
    assert sorted(p.name for p in extracted) == ["evil.txt", "notes.txt"]
    assert all(p.parent == tmp_path / "out" for p in extracted)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_a_zip_bomb_or_an_oversized_archive_is_refused_and_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "zip_max_mb", 1)
    bomb = tmp_path / "bomb.zip"
    bomb.write_bytes(_zip({"zeros.txt": b"\0" * (3 * 1024 * 1024)}))   # a few KB on disk, 3 MB unpacked
    with pytest.raises(ValueError, match="more than 1 MB"):
        expand_zip(bomb, tmp_path / "out")
    assert not (tmp_path / "out").exists()

    monkeypatch.setattr(settings, "zip_max_files", 2)
    many = tmp_path / "many.zip"
    many.write_bytes(_zip({f"{i}.txt": b"a" for i in range(3)}))
    with pytest.raises(ValueError, match="more than 2 files"):
        expand_zip(many, tmp_path / "out2")

    batch = create_batch()                                              # an upload reports it and keeps no archive
    result = save_upload(batch, "many.zip", many.read_bytes())
    assert result["status"] == "error" and "more than 2 files" in result["message"]
    assert list(batch.iterdir()) == []


def test_an_oversized_upload_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "upload_max_mb", 1)
    client = TestClient(app)
    big = client.post("/api/ingest/upload", files={"files": ("big.txt", b"a" * (1024 * 1024 + 1))})
    assert big.status_code == 413
    assert not any(Path(settings.uploads_dir).glob("*/*"))
    assert client.post("/api/ingest/upload", files={"files": ("ok.txt", b"hello")}).status_code == 200


def test_with_a_password_set_everything_but_health_needs_it(monkeypatch):
    monkeypatch.setattr(settings, "auth_password", "test-only-secret")
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    for path in ("/", "/review", "/api/library", "/api/entities"):
        response = client.get(path)
        assert response.status_code == 401 and "Basic" in response.headers["www-authenticate"]
    assert client.get("/api/library", auth=("anyone", "wrong")).status_code == 401
    assert client.get("/api/library", auth=("anyone", "test-only-secret")).status_code == 200


def test_without_a_password_the_app_is_open():
    assert TestClient(app).get("/api/library").status_code == 200
