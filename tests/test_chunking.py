from __future__ import annotations

import re
from pathlib import Path

from nlp_tda.embed import embeddings
from nlp_tda.ingest.chunking import TokenBudget, chunk_document
from nlp_tda.ingest.parsers import ParsedDocument


class WordTokenizer:
    """One token per word, with the character offsets a fast tokenizer reports."""

    def __call__(self, text: str, **_: object) -> dict:
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    def num_special_tokens_to_add(self) -> int:
        return 2


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(path=Path("notes.txt"), title="notes", text=text, mime_hint="txt", content_hash="h")


def test_token_chunks_fit_the_model_and_keep_the_original_text():
    text = " ".join(f"Wort{i}" for i in range(25))
    chunks = chunk_document(_doc(text), tokens=TokenBudget(WordTokenizer(), max_tokens=10, overlap=2))
    assert [len(c.text.split()) for c in chunks] == [10, 10, 9]            # no chunk over the limit
    assert chunks[1].text.split()[:2] == chunks[0].text.split()[-2:]       # overlap of 2 tokens
    assert chunks[-1].text.endswith("Wort24")                              # nothing dropped at the end
    for c in chunks:
        start, end = map(int, c.span_ref.split(":")[1].split("-"))
        assert text[start:end] == c.text                                   # span_ref points at the exact text


def test_without_a_token_budget_chunks_are_sized_in_characters():
    chunks = chunk_document(_doc("x" * 2000), chunk_size=800, overlap=120)
    assert [len(c.text) for c in chunks] == [800, 800, 640]


def test_token_budget_follows_the_model_limit(monkeypatch):
    class Model:
        max_seq_length = 128
        tokenizer = WordTokenizer()

    monkeypatch.setattr(embeddings.settings, "use_hash_embeddings", False)
    monkeypatch.setattr(embeddings.settings, "chunk_tokens", 0)
    monkeypatch.setattr(embeddings, "_load_model", lambda: Model())
    budget = embeddings.token_budget()
    assert budget.max_tokens == 126 and budget.overlap == 24
    assert embeddings.embedding_mode() == "model"
    assert embeddings.token_budget(force_hash=True) is None and embeddings.embedding_mode(force_hash=True) == "hash"
