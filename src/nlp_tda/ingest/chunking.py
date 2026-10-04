from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from nlp_tda.ingest.parsers import ParsedDocument


@dataclass
class Chunk:
    id: str
    artifact_id: str
    text: str
    span_ref: str
    index: int


@dataclass(frozen=True)
class TokenBudget:
    """What the embedding model can read in one input: its tokenizer and token limit."""

    tokenizer: Any
    max_tokens: int
    overlap: int


def artifact_id_for(doc: ParsedDocument) -> str:
    # Include relative-ish path so unzipped copies with same basename don't collide
    # when content differs; identical content still shares an id via content_hash.
    return hashlib.sha1(f"{doc.path.as_posix()}:{doc.content_hash}".encode()).hexdigest()[:16]


def _char_spans(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    step = max(chunk_size - overlap, 1)
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        spans.append((start, end))
        if end >= n:
            break
        start += step
    return spans


def _token_spans(text: str, budget: TokenBudget) -> list[tuple[int, int]]:
    """Character spans of at most ``max_tokens`` tokens each, so no chunk is cut off when embedded."""
    offsets = budget.tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False
    )["offset_mapping"]
    spans: list[tuple[int, int]] = []
    step = max(budget.max_tokens - budget.overlap, 1)
    for first in range(0, len(offsets), step):
        window = offsets[first : first + budget.max_tokens]
        spans.append((window[0][0], window[-1][1]))
        if first + budget.max_tokens >= len(offsets):
            break
    return spans


def chunk_document(
    doc: ParsedDocument,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    tokens: TokenBudget | None = None,
) -> list[Chunk]:
    """Cut a document into overlapping chunks of its own text.

    With a ``tokens`` budget the chunks are sized in the embedding model's tokens; without one
    (hash embeddings, no model) they are ``chunk_size`` characters.
    """
    art_id = artifact_id_for(doc)
    text = doc.text.strip()
    if not text:
        return []

    spans = _token_spans(text, tokens) if tokens is not None else _char_spans(text, chunk_size, overlap)
    chunks: list[Chunk] = []
    for start, end in spans:
        piece = text[start:end].strip()
        if not piece:
            continue
        index = len(chunks)
        cid = hashlib.sha1(f"{art_id}:{index}:{start}:{end}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                id=cid,
                artifact_id=art_id,
                text=piece,
                span_ref=f"{doc.path.name}:{start}-{end}",
                index=index,
            )
        )
    return chunks
