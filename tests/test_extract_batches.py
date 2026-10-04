from __future__ import annotations

import numpy as np
import pytest

from nlp_tda import ask
from nlp_tda.config import settings
from nlp_tda.extract.ollama_client import OllamaClient, extract_entities


@pytest.fixture
def fake_llm(monkeypatch):
    """An LLM that reports one requirement per chunk it is shown, plus the same client every call."""
    calls: list[str] = []

    def chat_json(self, system: str, user: str) -> dict:
        calls.append(user)
        shown = [w for w in user.split() if w.startswith("chunk-")]
        if any(w == "chunk-7" for w in shown):
            raise RuntimeError("model timed out")
        return {
            "clients": [{"name": "Nordwind  Logistics GmbH"}],
            "requirements": [{"statement": f"Requirement from {w}"} for w in shown],
        }

    monkeypatch.setattr(OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(OllamaClient, "chat_json", chat_json)
    monkeypatch.setattr(settings, "extract_batch_chunks", 3)
    monkeypatch.setattr(settings, "extract_max_batches", 0)
    return calls


def test_every_chunk_is_extracted_not_only_the_first(fake_llm):
    texts = [f"chunk-{i} text" for i in range(20) if i != 7]
    bundle, mode, read = extract_entities(texts, [])
    assert mode == "ollama" and read == 19
    assert len(fake_llm) == 7                                    # 19 chunks, 3 per call
    assert {r.statement for r in bundle.requirements} == {f"Requirement from chunk-{i}" for i in range(20) if i != 7}
    assert len(bundle.clients) == 1                              # named in every batch, kept once


def test_a_failed_batch_is_skipped_and_reported(fake_llm):
    texts = [f"chunk-{i} text" for i in range(9)]
    bundle, mode, read = extract_entities(texts, [])
    assert mode == "ollama-partial" and read == 9
    assert {r.statement for r in bundle.requirements} == {f"Requirement from chunk-{i}" for i in (0, 1, 2, 3, 4, 5)}
    assert all(c.name != "Unknown Client" for c in bundle.clients)  # no mock records mixed in


def test_batch_limit_reads_batches_spread_over_the_pack(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "extract_max_batches", 2)
    texts = [f"chunk-{i + 10} text" for i in range(12)]          # 4 batches of 3
    bundle, mode, read = extract_entities(texts, [])
    assert read == 6 and len(fake_llm) == 2
    assert {r.statement for r in bundle.requirements} == {
        f"Requirement from chunk-{i}" for i in (10, 11, 12, 16, 17, 18)
    }


def test_questions_use_the_configured_embedding_mode(monkeypatch):
    seen: dict = {}

    class Store:
        def count(self) -> int:
            return 1

        def query(self, vector, n_results):
            return {"documents": [["excerpt"]], "metadatas": [[{"span_ref": "a.txt:0-7"}]]}

    def embed_texts(texts, *, force_hash=None):
        seen["force_hash"] = force_hash
        return np.zeros((1, 384), dtype=np.float32)

    monkeypatch.setattr(ask, "ChunkStore", Store)
    monkeypatch.setattr(ask, "embed_texts", embed_texts)
    assert ask._retrieve_contexts("What are the requirements?")[0]["text"] == "excerpt"
    assert seen["force_hash"] is None                            # not forced to hash embeddings

