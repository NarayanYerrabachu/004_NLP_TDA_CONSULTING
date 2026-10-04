from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nlp_tda.config import settings
from nlp_tda.db import get_session, reset_db_for_tests
from nlp_tda.extract.ollama_client import OllamaClient
from nlp_tda.models import ProposedEntity
from nlp_tda.pipeline import run_pipeline
from nlp_tda.records import apply_edits

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_engagement"


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch):
    for name, path in (("data_dir", "data"), ("sqlite_path", "data/master.db"), ("chroma_path", "data/chroma"),
                       ("uploads_dir", "data/uploads"), ("exports_dir", "data/exports")):
        monkeypatch.setattr(settings, name, tmp_path / path)
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "force_mock_llm", True)
    monkeypatch.setattr(settings, "use_hash_embeddings", True)
    reset_db_for_tests()
    yield
    reset_db_for_tests()


def _rows() -> list[tuple[str, str, str]]:
    session = get_session()
    try:
        return sorted((r.entity_type, r.title, r.review_status) for r in session.query(ProposedEntity))
    finally:
        session.close()


def _review(title: str, status: str, edits: dict | None = None) -> None:
    session = get_session()
    try:
        row = session.query(ProposedEntity).filter(ProposedEntity.title == title).one()
        if edits:
            apply_edits(row, edits)
        row.review_status = status
        session.commit()
    finally:
        session.close()


def test_running_the_same_pack_again_does_not_duplicate_drafts():
    first = run_pipeline(FIXTURES)
    before = _rows()
    second = run_pipeline(FIXTURES)
    assert _rows() == before
    assert second["entities"] == first["entities"] == second["drafts_replaced"]
    assert first["drafts_replaced"] == 0


def test_reviewed_records_survive_a_rerun_and_are_not_proposed_again():
    run_pipeline(FIXTURES)
    _review("Marcus Lee", "accepted")
    _review("Target Operating Model blueprint", "rejected")
    _review("Dr. Lena Hartmann", "edited", {"name": "Dr. Lena Hartmann-Voss"})

    result = run_pipeline(FIXTURES)
    rows = _rows()
    assert result["already_reviewed"] == 3
    assert ("person", "Marcus Lee", "accepted") in rows
    assert ("deliverable", "Target Operating Model blueprint", "rejected") in rows
    assert ("person", "Dr. Lena Hartmann-Voss", "edited") in rows
    titles = [title for _, title, _ in rows]
    assert titles.count("Marcus Lee") == 1 and titles.count("Target Operating Model blueprint") == 1
    assert "Dr. Lena Hartmann" not in titles                     # the renamed record is not proposed again


def test_a_pack_uploaded_again_to_another_folder_replaces_its_drafts(tmp_path, monkeypatch):
    def chat_json(self, system: str, user: str) -> dict:
        return {"people": [{"name": "Lena Hartmann"}], "findings": [{"statement": "A finding nobody wrote down"}]}

    monkeypatch.setattr(settings, "force_mock_llm", False)
    monkeypatch.setattr(OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(OllamaClient, "chat_json", chat_json)
    first_upload, second_upload = tmp_path / "batch1", tmp_path / "batch2"
    first_upload.mkdir()
    (first_upload / "kickoff.txt").write_text("Sponsor: Lena Hartmann. Kickoff on Monday.")
    shutil.copytree(first_upload, second_upload)

    run_pipeline(first_upload)
    run_pipeline(second_upload)
    people = [r for r in _rows() if r[0] == "person"]
    assert people == [("person", "Lena Hartmann", "proposed")]   # found by document content, not by folder
