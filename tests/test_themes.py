from __future__ import annotations

import numpy as np

from nlp_tda.ingest.chunking import Chunk
from nlp_tda.tda.persistence import discover_themes

WORDS = {"invoice": "invoice payment supplier", "warehouse": "warehouse forklift picking", "hiring": "hiring onboarding training"}


def _pack(per_topic: int = 6, spread: float = 0.05, seed: int = 0) -> tuple[list[Chunk], np.ndarray]:
    """Chunks on three topics: each topic has its own direction in embedding space and its own words."""
    rng = np.random.default_rng(seed)
    chunks, vectors = [], []
    for axis, (topic, words) in enumerate(WORDS.items()):
        for i in range(per_topic):
            vector = np.zeros(8)
            vector[axis] = 1.0
            vectors.append(vector + rng.normal(0, spread, 8))
            chunks.append(Chunk(id=f"{topic}-{i}", artifact_id=topic, text=f"The {words} process and the team", span_ref=topic, index=i))
    return chunks, np.array(vectors)


def test_themes_follow_the_topics_and_are_named_by_their_words():
    chunks, vectors = _pack()
    themes = discover_themes(chunks, vectors)
    assert len(themes) == 3 and not any(t.outlier for t in themes)
    for theme in themes:
        topics = {cid.split("-")[0] for cid in theme.member_chunk_ids}
        assert len(topics) == 1 and len(theme.member_chunk_ids) == 6           # one topic per theme, nothing lost
        assert set(theme.label.split(" · ")) == set(WORDS[topics.pop()].split())  # its own words, not the shared ones
        assert 0.5 < theme.stability_score <= 1.0                               # tight and far from the others


def test_a_tighter_theme_is_more_stable_than_a_loose_one():
    rng = np.random.default_rng(1)
    tight = np.array([1.0, 0, 0]) + rng.normal(0, 0.01, (6, 3))
    loose = np.array([0, 1.0, 0]) + rng.normal(0, 0.25, (6, 3))
    chunks = [Chunk(id=f"{name}-{i}", artifact_id=name, text=f"{name} text", span_ref=name, index=i)
              for name in ("tight", "loose") for i in range(6)]
    themes = {t.member_chunk_ids[0].split("-")[0]: t for t in discover_themes(chunks, np.vstack([tight, loose]))}
    assert themes["tight"].stability_score > themes["loose"].stability_score


def test_a_chunk_unlike_the_rest_is_an_outlier_not_a_theme():
    chunks, vectors = _pack()
    odd = np.zeros(8)
    odd[7] = 1.0
    chunks.append(Chunk(id="odd-0", artifact_id="odd", text="cafeteria menu coffee", span_ref="odd", index=0))
    themes = discover_themes(chunks, np.vstack([vectors, odd]))
    outliers = [t for t in themes if t.outlier]
    assert len(themes) == 4 and len(outliers) == 1
    assert outliers[0].member_chunk_ids == ["odd-0"] and outliers[0].label.startswith("Outliers: ")
    assert outliers[0].stability_score == 0.0


def test_without_structure_the_pack_is_one_theme():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(30, 64))                                         # no groups at all
    chunks = [Chunk(id=str(i), artifact_id="a", text="notes about the project", span_ref="a", index=i) for i in range(30)]
    themes = discover_themes(chunks, vectors)
    assert len(themes) == 1 and len(themes[0].member_chunk_ids) == 30
    assert themes[0].stability_score == 0.0 and "no separate themes" in themes[0].persistence_summary


def test_every_chunk_gets_a_theme_when_only_a_sample_is_clustered():
    chunks, vectors = _pack(per_topic=30)
    themes = discover_themes(chunks, vectors, max_points=20)
    assert sorted(len(t.member_chunk_ids) for t in themes) == [30, 30, 30]
    assert all(len({cid.split("-")[0] for cid in t.member_chunk_ids}) == 1 for t in themes)


def test_tiny_packs_do_not_fail():
    one = [Chunk(id="0", artifact_id="a", text="kickoff notes", span_ref="a", index=0)]
    assert [t.member_chunk_ids for t in discover_themes(one, np.ones((1, 4)))] == [["0"]]
    assert discover_themes([], np.zeros((0, 4))) == []
