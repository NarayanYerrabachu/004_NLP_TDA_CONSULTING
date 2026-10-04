from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from nlp_tda.config import settings
from nlp_tda.ingest.chunking import Chunk


class ChunkStore:
    def __init__(self, collection_name: str = "chunks") -> None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        run_id: str,
    ) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "artifact_id": c.artifact_id,
                    "span_ref": c.span_ref,
                    "index": c.index,
                    "run_id": run_id,
                }
                for c in chunks
            ],
        )

    def query(self, embedding: list[float], n_results: int = 5) -> dict[str, Any]:
        return self._collection.query(query_embeddings=[embedding], n_results=n_results)

    def count(self) -> int:
        return self._collection.count()
