from __future__ import annotations

from pathlib import Path

import pytest

from nlp_tda.config import settings
from nlp_tda.db import reset_db_for_tests
from nlp_tda.embed.embeddings import embed_chunks
from nlp_tda.ingest.chunking import chunk_document
from nlp_tda.ingest.parsers import parse_directory, parse_file, probe_filename
from nlp_tda.ingest.uploads import create_batch, save_upload
from nlp_tda.pipeline import run_pipeline
from nlp_tda.tda.persistence import discover_themes


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_engagement"


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path):
    settings.data_dir = tmp_path / "data"
    settings.sqlite_path = tmp_path / "data" / "master.db"
    settings.chroma_path = tmp_path / "data" / "chroma"
    settings.uploads_dir = tmp_path / "data" / "uploads"
    settings.exports_dir = tmp_path / "data" / "exports"
    settings.project_root = tmp_path
    settings.force_mock_llm = True
    settings.use_hash_embeddings = True
    reset_db_for_tests()
    yield
    reset_db_for_tests()


def test_parsers_cover_mvp_formats():
    docs = parse_directory(FIXTURES)
    hints = {d.mime_hint for d in docs}
    assert {"pdf", "docx", "pptx", "md", "txt", "csv", "xlsx", "eml"} <= hints
    assert all(d.text.strip() or d.warning for d in docs)


def test_eml_and_csv_parse():
    eml = parse_file(FIXTURES / "09_kickoff_email.eml")
    assert "Nordwind" in eml.text
    assert eml.mime_hint == "eml"
    csv_doc = parse_file(FIXTURES / "07_stakeholders.csv")
    assert "Hartmann" in csv_doc.text


def test_probe_later_images():
    p = probe_filename("scan.png")
    assert p.status == "later"


def test_upload_zip_expands(tmp_path):
    settings.data_dir = tmp_path / "data"
    settings.uploads_dir = tmp_path / "data" / "uploads"
    settings.exports_dir = tmp_path / "data" / "exports"
    settings.project_root = tmp_path
    batch = create_batch()
    zip_bytes = (FIXTURES / "10_pack_slice.zip").read_bytes()
    result = save_upload(batch, "10_pack_slice.zip", zip_bytes)
    assert result["status"] == "expanded"
    assert result["children"]
    assert any(c["status"] in {"parsed", "parsed_with_warning"} for c in result["children"])


def test_tda_returns_themes():
    docs = parse_directory(FIXTURES)
    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d, chunk_size=400, overlap=40))
    emb = embed_chunks(chunks, force_hash=True)
    themes = discover_themes(chunks, emb, max_points=80)
    assert themes
    assert any(t.member_chunk_ids for t in themes)


def test_pipeline_smoke_synthetic():
    result = run_pipeline(FIXTURES, force_hash_embeddings=True)
    assert result["documents"] >= 5
    assert result["chunks"] > 0
    assert result["themes"] >= 1
    assert result["entities"] >= 5
    assert result["llm_mode"] in {"mock", "mock-fallback"}
