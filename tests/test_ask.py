from nlp_tda.ask import handle_command


def test_ask_help():
    out = handle_command("help", lang="en")
    assert out["kind"] == "help"
    assert out["lines"]


def test_ask_status(tmp_path, monkeypatch):
    from nlp_tda.config import settings
    from nlp_tda.db import reset_db_for_tests

    settings.data_dir = tmp_path / "data"
    settings.sqlite_path = tmp_path / "data" / "master.db"
    settings.chroma_path = tmp_path / "data" / "chroma"
    settings.uploads_dir = tmp_path / "data" / "uploads"
    settings.exports_dir = tmp_path / "data" / "exports"
    settings.project_root = tmp_path
    settings.force_mock_llm = True
    settings.use_hash_embeddings = True
    reset_db_for_tests()
    out = handle_command("status", lang="en")
    assert out["kind"] == "status"
    assert "documents" in out
