from nlp_tda.config import settings


def test_settings_defaults():
    assert settings.port == 38417
    assert "MiniLM" in settings.embedding_model or "bge" in settings.embedding_model.lower()
