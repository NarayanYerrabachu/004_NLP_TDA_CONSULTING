from __future__ import annotations

import hashlib
from dataclasses import dataclass

from nlp_tda.ingest.parsers import ParsedDocument


@dataclass
class Chunk:
    id: str
    artifact_id: str
    text: str
    span_ref: str
    index: int


def artifact_id_for(doc: ParsedDocument) -> str:
    # Include relative-ish path so unzipped copies with same basename don't collide
    # when content differs; identical content still shares an id via content_hash.
    return hashlib.sha1(f"{doc.path.as_posix()}:{doc.content_hash}".encode()).hexdigest()[:16]


def chunk_document(
    doc: ParsedDocument,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    art_id = artifact_id_for(doc)
    text = doc.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    n = len(text)
    step = max(chunk_size - overlap, 1)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
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
            index += 1
        if end >= n:
            break
        start += step
    return chunks
