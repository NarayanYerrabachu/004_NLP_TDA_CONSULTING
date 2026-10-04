from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import numpy as np

from nlp_tda.config import settings
from nlp_tda.ingest.chunking import Chunk, TokenBudget

log = logging.getLogger(__name__)


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


def _model(force_hash: bool | None):
    """The embedding model, or None when hash embeddings are asked for or the model cannot load."""
    if settings.use_hash_embeddings if force_hash is None else force_hash:
        return None
    try:
        return _load_model()
    except Exception as exc:
        # Network / model download failure → still allow pipeline demo
        log.warning("Embedding model %s unavailable, using hash embeddings: %s", settings.embedding_model, exc)
        return None


def embedding_mode(*, force_hash: bool | None = None) -> str:
    """ "model" when texts are embedded by the sentence-transformer, "hash" when by the fallback."""
    return "hash" if _model(force_hash) is None else "model"


def token_budget(*, force_hash: bool | None = None) -> TokenBudget | None:
    """How much text the embedding model reads per input; None with hash embeddings."""
    model = _model(force_hash)
    if model is None:
        return None
    limit = model.max_seq_length - model.tokenizer.num_special_tokens_to_add()
    max_tokens = min(settings.chunk_tokens, limit) if settings.chunk_tokens > 0 else limit
    return TokenBudget(model.tokenizer, max_tokens, min(settings.chunk_token_overlap, max_tokens // 2))


def embed_texts(texts: list[str], *, force_hash: bool | None = None) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _model(force_hash)
    if model is None:
        return _hash_embed(texts)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_chunks(chunks: list[Chunk], *, force_hash: bool | None = None) -> np.ndarray:
    return embed_texts([c.text for c in chunks], force_hash=force_hash)
