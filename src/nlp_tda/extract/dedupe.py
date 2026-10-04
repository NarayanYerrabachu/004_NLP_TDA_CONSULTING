from __future__ import annotations

from nlp_tda.config import settings
from nlp_tda.embed.embeddings import embed_texts, embedding_mode
from nlp_tda.models import ExtractionBundle


def merge_near_duplicates(bundle: ExtractionBundle) -> int:
    """Requirements and findings that say the same thing in other words become one record.

    Two statements are the same when their embeddings are at least ``settings.dedupe_similarity``
    alike (this also joins a German and an English wording). The longest wording is kept and the
    others are listed in its ``also_stated``, so no wording is lost. Names (clients, people,
    engagements, deliverables) are not compared this way: different names embed alike.
    Returns how many records were merged away; 0 without the embedding model.
    """
    threshold = settings.dedupe_similarity
    if threshold <= 0 or embedding_mode() != "model":
        return 0
    merged_away = 0
    for field in ("requirements", "findings"):
        items = getattr(bundle, field)
        if len(items) < 2:
            continue
        vectors = embed_texts([item.statement for item in items])
        kept: list[int] = []
        for i, item in enumerate(items):
            slot = next((s for s, k in enumerate(kept) if float(vectors[i] @ vectors[k]) >= threshold), None)
            if slot is None:
                kept.append(i)
                continue
            merged_away += 1
            keeper = items[kept[slot]]
            if len(item.statement) > len(keeper.statement):
                item.also_stated = [*keeper.also_stated, keeper.statement, *item.also_stated]
                kept[slot] = i
            else:
                keeper.also_stated.append(item.statement)
        setattr(bundle, field, [items[k] for k in kept])
    return merged_away
