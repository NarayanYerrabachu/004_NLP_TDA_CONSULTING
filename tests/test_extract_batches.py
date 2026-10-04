from __future__ import annotations

import numpy as np
import pytest

from nlp_tda import ask
from nlp_tda.config import settings
from nlp_tda.extract import dedupe
from nlp_tda.extract.ollama_client import LLMUnavailable, OllamaClient, extract_entities
from nlp_tda.ingest.chunking import Chunk
from nlp_tda.models import ExtractionBundle


def _chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", artifact_id=f"doc{i}", text=t, span_ref=f"doc{i}.txt:0-{len(t)}", index=0)
        for i, t in enumerate(texts)
    ]


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
    monkeypatch.setattr(settings, "use_hash_embeddings", True)
    monkeypatch.setattr(settings, "force_mock_llm", False)
    return calls


def test_every_chunk_is_extracted_not_only_the_first(fake_llm):
    texts = [f"chunk-{i} text" for i in range(20) if i != 7]
    bundle, mode, read = extract_entities(_chunks(texts), [])
    assert mode == "ollama" and read == 19
    assert len(fake_llm) == 7                                    # 19 chunks, 3 per call
    assert {r.statement for r in bundle.requirements} == {f"Requirement from chunk-{i}" for i in range(20) if i != 7}
    assert len(bundle.clients) == 1                              # named in every batch, kept once
    assert bundle.clients[0].mentions == 7


def test_a_failed_batch_is_skipped_and_reported(fake_llm):
    texts = [f"chunk-{i} text" for i in range(9)]
    bundle, mode, read = extract_entities(_chunks(texts), [])
    assert mode == "ollama-partial" and read == 9
    assert {r.statement for r in bundle.requirements} == {f"Requirement from chunk-{i}" for i in (0, 1, 2, 3, 4, 5)}
    assert all(c.name != "Unknown Client" for c in bundle.clients)  # no mock records mixed in


def test_batch_limit_reads_batches_spread_over_the_pack(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "extract_max_batches", 2)
    texts = [f"chunk-{i + 10} text" for i in range(12)]          # 4 batches of 3
    bundle, mode, read = extract_entities(_chunks(texts), [])
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



def test_records_point_at_the_chunk_they_were_read_from(monkeypatch):
    def chat_json(self, system: str, user: str) -> dict:
        return {
            "people": [{"name": "Lena Hartmann", "chunk": 1}],                      # named in excerpt 2: the text wins
            "findings": [{"statement": "Planning is split over three tools", "chunk": "[3]"}],   # paraphrase: citation
            "requirements": [{"statement": "Something the model made up", "chunk": 9}],          # no such excerpt
        }

    monkeypatch.setattr(OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(OllamaClient, "chat_json", chat_json)
    monkeypatch.setattr(settings, "use_hash_embeddings", True)
    monkeypatch.setattr(settings, "force_mock_llm", False)
    chunks = _chunks(["Kickoff agenda", "Sponsor: Dr. Lena  Hartmann (Nordwind)", "Three planning tools are in use"])
    bundle, _, _ = extract_entities(chunks, [])
    assert (bundle.people[0].source_artifact_id, bundle.people[0].span_ref) == ("doc1", chunks[1].span_ref)
    assert (bundle.findings[0].source_artifact_id, bundle.findings[0].span_ref) == ("doc2", chunks[2].span_ref)
    assert bundle.requirements[0].source_artifact_id is None and bundle.requirements[0].span_ref is None
    assert "chunk" not in bundle.people[0].model_dump()                             # the citation is not stored
    # confidence follows the evidence, not the LLM's own number
    assert (bundle.people[0].evidence, bundle.people[0].confidence) == ("verbatim", 0.9)
    assert (bundle.findings[0].evidence, bundle.findings[0].confidence) == ("cited", 0.6)
    assert (bundle.requirements[0].evidence, bundle.requirements[0].confidence) == ("none", 0.3)


def test_the_llms_own_confidence_is_ignored_and_repeats_raise_it(fake_llm, monkeypatch):
    def chat_json(self, system: str, user: str) -> dict:
        return {"clients": [{"name": "Nordwind", "confidence": 95}]}      # out of range: must not fail the call

    monkeypatch.setattr(OllamaClient, "chat_json", chat_json)
    bundle, mode, _ = extract_entities(_chunks(["Nordwind kickoff", "notes", "more", "Nordwind again"]), [])
    assert mode == "ollama"
    assert (bundle.clients[0].mentions, bundle.clients[0].confidence) == (2, 1.0)   # verbatim 0.9 + repeated 0.1


def test_an_unreachable_llm_fails_the_run_instead_of_returning_mock_records(monkeypatch):
    monkeypatch.setattr(settings, "force_mock_llm", False)
    monkeypatch.setattr(OllamaClient, "available", lambda self: False)
    with pytest.raises(LLMUnavailable, match="not reachable"):
        extract_entities(_chunks(["Nordwind kickoff notes"]), [])


def test_an_llm_that_answers_no_batch_fails_the_run(fake_llm, monkeypatch):
    def chat_json(self, system: str, user: str) -> dict:
        raise RuntimeError("model 'qwen' not found")

    monkeypatch.setattr(OllamaClient, "chat_json", chat_json)
    with pytest.raises(LLMUnavailable, match="not found"):
        extract_entities(_chunks(["Nordwind kickoff notes"]), [])


def test_mock_mode_returns_the_fixture_records(monkeypatch):
    monkeypatch.setattr(settings, "force_mock_llm", True)
    bundle, mode, read = extract_entities(_chunks(["Nordwind kickoff notes"]), [])
    assert mode == "mock" and read == 1 and bundle.clients[0].name == "Nordwind Logistics GmbH"


def test_the_same_statement_in_other_words_is_one_record(monkeypatch):
    vectors = {
        "Fragmented planning tools": [1.0, 0.0, 0.0],
        "Planning tools are fragmented across three regional hubs": [0.9, 0.436, 0.0],   # cosine 0.9 with the first
        "Invoice reconciliation is manual": [0.0, 0.0, 1.0],
    }
    monkeypatch.setattr(dedupe, "embedding_mode", lambda: "model")
    monkeypatch.setattr(dedupe, "embed_texts", lambda texts: np.array([vectors[t] for t in texts]))
    monkeypatch.setattr(settings, "dedupe_similarity", 0.6)
    bundle = ExtractionBundle.model_validate({"findings": [{"statement": s} for s in vectors]})
    assert dedupe.merge_near_duplicates(bundle) == 1
    assert [f.statement for f in bundle.findings] == [
        "Planning tools are fragmented across three regional hubs",                # the longer wording is kept
        "Invoice reconciliation is manual",
    ]
    assert bundle.findings[0].also_stated == ["Fragmented planning tools"]          # the other wording is not lost


def test_without_the_embedding_model_nothing_is_merged(monkeypatch):
    monkeypatch.setattr(dedupe, "embedding_mode", lambda: "hash")
    bundle = ExtractionBundle.model_validate({"findings": [{"statement": "a"}, {"statement": "a b"}]})
    assert dedupe.merge_near_duplicates(bundle) == 0 and len(bundle.findings) == 2
