"""TDA structure discovery via ripser (Vietoris–Rips persistence).

Why ripser for MVP: lightest VR persistence stack that installs cleanly via pip,
avoids heavy GUDHI/giotto-tda native builds, and is enough to surface stable
components / outliers on a PCA-reduced embedding cloud for small engagement packs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ripser import ripser
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from nlp_tda.config import settings
from nlp_tda.ingest.chunking import Chunk


@dataclass
class ThemeCandidate:
    label: str
    stability_score: float
    member_chunk_ids: list[str]
    persistence_summary: str
    outlier: bool = False


def discover_themes(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    *,
    pca_dims: int | None = None,
    max_points: int | None = None,
) -> list[ThemeCandidate]:
    if not chunks or embeddings.size == 0:
        return []

    pca_dims = pca_dims or settings.tda_pca_dims
    max_points = max_points or settings.tda_max_points

    n = embeddings.shape[0]
    idx = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))

    sample = embeddings[idx]
    dims = min(pca_dims, sample.shape[0] - 1, sample.shape[1])
    if dims < 2:
        # Degenerate: one theme with all members
        return [
            ThemeCandidate(
                label="Theme 1",
                stability_score=0.1,
                member_chunk_ids=[c.id for c in chunks],
                persistence_summary="insufficient points for VR filtration",
            )
        ]

    reduced = PCA(n_components=dims, random_state=42).fit_transform(sample)
    # Finite metric for ripser
    dists = pairwise_distances(reduced, metric="euclidean")
    result = ripser(dists, distance_matrix=True, maxdim=1)
    diagrams = result["dgms"]

    h0 = diagrams[0]
    # Finite H0 death times ≈ merge scale; longer life ≈ more stable component
    finite_h0 = h0[np.isfinite(h0[:, 1])]
    if len(finite_h0) == 0:
        n_clusters = 1
        lifetimes = np.array([1.0])
    else:
        lifetimes = finite_h0[:, 1] - finite_h0[:, 0]
        # Number of relatively persistent components (exclude trivial noise)
        median = float(np.median(lifetimes)) if len(lifetimes) else 0.0
        persistent = lifetimes[lifetimes >= max(median, 1e-6)]
        n_clusters = int(np.clip(len(persistent), 1, min(8, sample.shape[0])))

    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="average")
    labels = clustering.fit_predict(reduced)

    # Map sample labels back; unsampled points assigned by nearest sampled neighbor
    full_labels = np.full(n, -1, dtype=int)
    full_labels[idx] = labels
    if n > max_points:
        for i in range(n):
            if full_labels[i] >= 0:
                continue
            d = np.linalg.norm(embeddings[i] - embeddings[idx], axis=1)
            full_labels[i] = labels[int(np.argmin(d))]

    # Outliers: points far from their cluster centroid in reduced space (sampled only)
    centroids = {k: reduced[labels == k].mean(axis=0) for k in range(n_clusters)}
    outlier_sample = set()
    for local_i, global_i in enumerate(idx):
        k = labels[local_i]
        dist = np.linalg.norm(reduced[local_i] - centroids[k])
        # Flag top distant points relatively
        outlier_sample.add((dist, global_i, k))
    # Mark the farthest ~10% as outliers
    ordered = sorted(outlier_sample, reverse=True)
    outlier_ids = {chunks[g].id for _, g, _ in ordered[: max(1, len(ordered) // 10)]}

    h1_count = 0
    if len(diagrams) > 1:
        h1 = diagrams[1]
        h1_count = int(np.sum(np.isfinite(h1[:, 1])))

    themes: list[ThemeCandidate] = []
    for k in range(n_clusters):
        members = [chunks[i].id for i in range(n) if full_labels[i] == k]
        if not members:
            continue
        # Stability proxy: normalized mean H0 lifetime vs cluster size
        stab = float(np.mean(lifetimes)) if len(lifetimes) else 0.0
        stab = min(1.0, stab / (stab + 1.0) + 0.05 * min(len(members), 10))
        themes.append(
            ThemeCandidate(
                label=f"Theme {k + 1}",
                stability_score=round(stab, 3),
                member_chunk_ids=members,
                persistence_summary=(
                    f"H0_finite={len(finite_h0)}; H1_finite={h1_count}; "
                    f"pca_dims={dims}; members={len(members)}"
                ),
                outlier=any(m in outlier_ids for m in members) and len(members) <= 2,
            )
        )

    # Dedicated outlier theme if sparse high-distance points exist
    sparse = [cid for cid in outlier_ids if all(cid not in t.member_chunk_ids or len(t.member_chunk_ids) > 2 for t in themes)]
    if sparse:
        themes.append(
            ThemeCandidate(
                label="Outliers",
                stability_score=0.2,
                member_chunk_ids=list(outlier_ids),
                persistence_summary="far-from-centroid points in PCA+VR pipeline",
                outlier=True,
            )
        )

    return themes
