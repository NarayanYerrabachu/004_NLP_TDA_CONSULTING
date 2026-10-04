from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from nlp_tda.config import settings
from nlp_tda.ingest.chunking import Chunk


def _hash_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic fallback embeddings for offline smoke tests (not semantic)."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vec = rng.normal(size=dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        vectors.append(vec)
    return np.vstack(vectors)


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], *, force_hash: bool | None = None) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    use_hash = settings.use_hash_embeddings if force_hash is None else force_hash
    if use_hash:
        return _hash_embed(texts)
    try:
        model = _load_model()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
    except Exception:
        # Network / model download failure → still allow pipeline demo
        return _hash_embed(texts)


def embed_chunks(chunks: list[Chunk], *, force_hash: bool | None = None) -> np.ndarray:
    return embed_texts([c.text for c in chunks], force_hash=force_hash)
