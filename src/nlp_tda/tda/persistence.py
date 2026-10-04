"""Themes: groups of chunks that are about the same thing, each with a readable label and a
measure of how distinct it is.

Grouping is average-linkage clustering of the chunk embeddings (cosine distance), with the number
of groups chosen by silhouette. When the best grouping separates the chunks poorly, the pack is
reported as one theme instead of inventing several.

Stability is the theme's H0 persistence in the Vietoris-Rips filtration of its chunks: the theme
is exactly one connected component from the scale where its chunks join up ("formed") until the
scale where it touches a chunk outside ("merged"). stability = 1 - formed / merged; 0 means the
theme is never a component of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_score

from nlp_tda.config import settings
from nlp_tda.ingest.chunking import Chunk

MAX_THEMES = 8
MIN_THEME_CHUNKS = 2  # a lone chunk is an outlier, not a theme
LABEL_TERMS = 3

_GERMAN_STOP_WORDS = frozenset(
    "aber alle als also am an auch auf aus bei bis da damit dann das dass dem den der des die dies diese "
    "dieser dieses doch durch ein eine einem einen einer eines er es für hat haben ich ihr ihre im in ist "
    "je kann kein keine man mit muss nach nicht noch nur oder pro sein sich sie sind so soll sowie über um "
    "und unter von vor war was wenn werden wie wir wird zu zum zur".split()
)
_STOP_WORDS = sorted(ENGLISH_STOP_WORDS | _GERMAN_STOP_WORDS)


@dataclass
class ThemeCandidate:
    label: str
    stability_score: float
    member_chunk_ids: list[str]
    persistence_summary: str
    outlier: bool = False


def _top_terms(groups: list[list[str]]) -> list[list[str]]:
    """Per group of texts, the words that set it apart from the other groups (TF-IDF over groups)."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words=_STOP_WORDS, token_pattern=r"(?u)\b[^\W\d_]{3,}\b", sublinear_tf=True
        )
        scores = vectorizer.fit_transform(["\n".join(texts) for texts in groups]).toarray()
    except ValueError:  # no words left after stop words
        return [[] for _ in groups]
    vocabulary = vectorizer.get_feature_names_out()
    return [[vocabulary[i] for i in np.argsort(-row, kind="stable")[:LABEL_TERMS] if row[i] > 0] for row in scores]


def _persistence(distances: np.ndarray, members: np.ndarray) -> tuple[float, float]:
    """(formed, merged) scales of a group: where its chunks become one component, and where that
    component first touches a chunk outside the group (inf when there is nothing outside)."""
    inside = distances[np.ix_(members, members)]
    formed = float(linkage(squareform(inside, checks=False), method="single")[:, 2].max()) if len(members) > 1 else 0.0
    outside = np.setdiff1d(np.arange(len(distances)), members)
    merged = float(distances[np.ix_(members, outside)].min()) if len(outside) else float("inf")
    return formed, merged


def _group(distances: np.ndarray) -> tuple[np.ndarray, float]:
    """Cluster labels and their silhouette; one group (silhouette 0) when nothing separates well."""
    n = len(distances)
    best: tuple[float, np.ndarray] | None = None
    for k in range(2, min(MAX_THEMES, n - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(distances)
        score = float(silhouette_score(distances, labels, metric="precomputed"))
        if best is None or score > best[0]:
            best = (score, labels)
    if best is None or best[0] < settings.theme_min_separation:
        return np.zeros(n, dtype=int), 0.0
    return best[1], best[0]


def discover_themes(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    *,
    max_points: int | None = None,
) -> list[ThemeCandidate]:
    if not chunks or embeddings.size == 0:
        return []

    n = embeddings.shape[0]
    sample = np.arange(n)
    max_points = max_points or settings.tda_max_points
    if n > max_points:
        sample = np.sort(np.random.default_rng(42).choice(n, size=max_points, replace=False))
    distances = pairwise_distances(embeddings[sample], metric="cosine")
    np.fill_diagonal(distances, 0.0)
    sample_labels, separation = _group(distances)

    # Chunks outside the sample join the group of their nearest sampled chunk.
    labels = np.empty(n, dtype=int)
    labels[sample] = sample_labels
    rest = np.setdiff1d(np.arange(n), sample)
    if len(rest):
        nearest = pairwise_distances(embeddings[rest], embeddings[sample], metric="cosine").argmin(axis=1)
        labels[rest] = sample_labels[nearest]

    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    themes = sorted((g for g in groups if len(g) >= MIN_THEME_CHUNKS or len(groups) == 1), key=len, reverse=True)
    lone = [int(i) for g in groups if len(g) < MIN_THEME_CHUNKS and len(groups) > 1 for i in g]

    terms = _top_terms([[chunks[i].text for i in g] for g in themes] + ([[chunks[i].text for i in lone]] if lone else []))
    position = {int(s): p for p, s in enumerate(sample)}
    out: list[ThemeCandidate] = []
    for number, (members, words) in enumerate(zip(themes, terms), start=1):
        sampled = np.array([position[int(i)] for i in members if int(i) in position])
        formed, merged = _persistence(distances, sampled)
        stability = 0.0 if merged == float("inf") else max(0.0, 1.0 - formed / merged) if merged > 0 else 0.0
        if len(themes) == 1 and not lone:
            summary = f"no separate themes found; {len(members)} chunks"
        else:
            summary = (
                f"one component from scale {formed:.2f} until it meets another chunk at {merged:.2f}; "
                f"{len(members)} chunks; grouping silhouette {separation:.2f}"
            )
        out.append(
            ThemeCandidate(
                label=" · ".join(words) or f"Theme {number}",
                stability_score=round(stability, 3),
                member_chunk_ids=[chunks[int(i)].id for i in members],
                persistence_summary=summary,
            )
        )
    if lone:
        words = terms[-1]
        out.append(
            ThemeCandidate(
                label="Outliers" + (f": {' · '.join(words)}" if words else ""),
                stability_score=0.0,
                member_chunk_ids=[chunks[i].id for i in lone],
                persistence_summary=f"{len(lone)} chunk(s) unlike any theme",
                outlier=True,
            )
        )
    return out
