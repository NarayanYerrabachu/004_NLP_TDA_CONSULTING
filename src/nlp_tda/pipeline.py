from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from nlp_tda.config import settings
from nlp_tda.db import get_session
from nlp_tda.embed.chroma_store import ChunkStore
from nlp_tda.embed.embeddings import embed_chunks, embedding_mode, token_budget
from nlp_tda.extract.ollama_client import extract_entities
from nlp_tda.ingest.chunking import Chunk, artifact_id_for, chunk_document
from nlp_tda.ingest.parsers import expand_zip, parse_directory
from nlp_tda.models import ArtifactRow, PipelineRun, ProposedEntity, ReviewStatus, ThemeRecord
from nlp_tda.records import save_run_records
from nlp_tda.tda.persistence import discover_themes


def _prepare_source(source: Path) -> Path:
    """If source contains .zip files, expand them alongside other docs for parsing."""
    zips = list(source.rglob("*.zip")) if source.is_dir() else []
    if not zips:
        return source
    for zp in zips:
        expand_dir = zp.parent / f"_unzipped_{zp.stem}"
        if not expand_dir.exists():
            expand_zip(zp, expand_dir)
    return source


def run_pipeline(
    source_dir: Path | None = None,
    *,
    force_hash_embeddings: bool | None = None,
    progress: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Run the whole pipeline on a folder. ``progress(stage, done, total)`` is told each stage."""
    report = progress or (lambda *_: None)
    report("parsing")
    source = Path(source_dir or settings.fixtures_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    source = _prepare_source(source)
    run_id = uuid.uuid4().hex[:12]
    docs = parse_directory(source)
    # Dedupe identical payloads (e.g. same file also inside a zip)
    seen_hash: set[str] = set()
    unique_docs = []
    for doc in docs:
        if doc.content_hash in seen_hash:
            continue
        seen_hash.add(doc.content_hash)
        unique_docs.append(doc)
    docs = unique_docs
    if not docs:
        raise ValueError(f"No supported documents in {source}")

    all_chunks: list[Chunk] = []
    artifact_rows: list[ArtifactRow] = []
    tokens = token_budget(force_hash=force_hash_embeddings)
    for doc in docs:
        art_id = artifact_id_for(doc)
        artifact_rows.append(
            ArtifactRow(
                id=art_id,
                title=doc.title,
                path=str(doc.path),
                content_hash=doc.content_hash,
                mime_hint=doc.mime_hint,
                text_preview=doc.text[:500],
                run_id=run_id,
            )
        )
        all_chunks.extend(
            chunk_document(
                doc, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap, tokens=tokens
            )
        )

    report("embedding")
    embeddings = embed_chunks(all_chunks, force_hash=force_hash_embeddings)
    store = ChunkStore()
    store.upsert_chunks(
        all_chunks,
        embeddings.tolist() if embeddings.size else [],
        run_id=run_id,
    )

    report("themes")
    themes = discover_themes(all_chunks, embeddings)
    theme_labels = [t.label for t in themes]

    # Prefer DE prompts if corpus looks German-heavy
    blob = "\n".join(c.text for c in all_chunks[:20]).lower()
    prefer_de = sum(1 for w in ("und", "der", "die", "anforderung", "mandat") if w in blob) >= 2

    bundle, llm_mode, chunks_extracted = extract_entities(
        all_chunks,
        theme_labels,
        prefer_de=prefer_de,
        on_batch=lambda done, total: report("extracting", done, total),
    )
    report("saving")

    session = get_session()
    try:
        for row in artifact_rows:
            session.merge(row)

        session.flush()
        records: list[ProposedEntity] = []

        def _add(entity_type: str, title: str, payload: dict, conf: float, art: str | None, span: str | None, lang: str | None):
            records.append(
                ProposedEntity(
                    entity_type=entity_type,
                    title=title,
                    payload=payload,
                    confidence=conf,
                    source_artifact_id=art,
                    span_ref=span,
                    review_status=ReviewStatus.proposed.value,
                    language=lang,
                    run_id=run_id,
                )
            )

        for item in bundle.clients:
            _add("client", item.name, item.model_dump(), item.confidence, item.source_artifact_id, item.span_ref, item.language)
        for item in bundle.engagements:
            _add("engagement", item.title, item.model_dump(), item.confidence, item.source_artifact_id, item.span_ref, item.language)
        for item in bundle.people:
            _add("person", item.name, item.model_dump(), item.confidence, item.source_artifact_id, item.span_ref, item.language)
        for item in bundle.requirements:
            _add("requirement", item.statement[:120], item.model_dump(), item.confidence, item.source_artifact_id, item.span_ref, item.language)
        for item in bundle.findings:
            _add("finding", item.statement[:120], item.model_dump(), item.confidence, item.source_artifact_id, item.span_ref, item.language)
        for item in bundle.deliverables:
            _add("deliverable", item.name, item.model_dump(), item.confidence, item.source_artifact_id, item.span_ref, item.language)

        for theme in themes:
            rec = ThemeRecord(
                label=theme.label,
                stability_score=theme.stability_score,
                member_chunk_ids=theme.member_chunk_ids,
                persistence_summary=theme.persistence_summary,
                confidence=theme.stability_score,
                language="mixed",
            )
            _add(
                "theme",
                theme.label,
                {**rec.model_dump(), "outlier": theme.outlier},
                theme.stability_score,
                None,
                None,
                "mixed",
            )

        saved = save_run_records(
            session,
            records,
            run_id=run_id,
            source_dir=str(source),
            content_hashes={doc.content_hash for doc in docs},
        )
        entity_count = saved["added"]

        session.add(
            PipelineRun(
                id=run_id,
                source_dir=str(source),
                status="completed",
                llm_mode=llm_mode,
                theme_count=len(themes),
                entity_count=entity_count,
                notes="SYNTHETIC fixtures OK" if "synthetic" in str(source) else "",
            )
        )
        session.commit()
    finally:
        session.close()

    return {
        "run_id": run_id,
        "documents": len(docs),
        "chunks": len(all_chunks),
        "themes": len(themes),
        "entities": entity_count,
        "drafts_replaced": saved["replaced"],
        "already_reviewed": saved["already_reviewed"],
        "llm_mode": llm_mode,
        "embedding_mode": embedding_mode(force_hash=force_hash_embeddings),
        "chunks_extracted": chunks_extracted,
        "source_dir": str(source),
        "vector_count": store.count(),
    }
