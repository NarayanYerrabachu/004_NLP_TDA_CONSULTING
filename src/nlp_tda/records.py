"""Saving a run's records without piling up duplicates.

Unreviewed drafts are derived from the documents and are replaced when the same documents are
read again. Accepted, edited and rejected records are a reviewer's decisions: they stay, and the
same record is not proposed again.
"""

from __future__ import annotations

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from nlp_tda.models import ArtifactRow, PipelineRun, ProposedEntity, ReviewStatus

# Payload key: the titles a reviewed record was proposed under before a reviewer renamed it.
PROPOSED_AS = "proposed_as"


def _key(entity_type: str, title: object) -> tuple[str, str]:
    return entity_type, " ".join(str(title).lower().split())


def apply_edits(row: ProposedEntity, edits: dict) -> None:
    """A reviewer's edits to a record. A renamed record remembers the title it was proposed under."""
    payload = dict(row.payload or {})
    payload.update(edits)
    for field in ("name", "title", "statement"):
        if field in edits:
            new_title = str(edits[field])[:120] if field == "statement" else str(edits[field])
            if _key("", new_title) != _key("", row.title):
                payload[PROPOSED_AS] = [*payload.get(PROPOSED_AS, []), row.title]
            row.title = new_title
            break
    row.payload = payload


def save_run_records(
    session: Session,
    records: list[ProposedEntity],
    *,
    run_id: str,
    source_dir: str,
    content_hashes: set[str],
) -> dict[str, int]:
    """Store ``records`` as this run's drafts. Returns counts: added, replaced, already_reviewed."""
    proposed = ReviewStatus.proposed.value

    # Earlier drafts of the same documents (by content, so a re-uploaded pack counts), and drafts
    # without a source document from earlier runs on the same folder (themes).
    same_documents = select(ArtifactRow.id).where(ArtifactRow.content_hash.in_(content_hashes))
    same_folder = select(PipelineRun.id).where(PipelineRun.source_dir == source_dir)
    replaced = (
        session.query(ProposedEntity)
        .filter(
            ProposedEntity.review_status == proposed,
            ProposedEntity.run_id != run_id,
            or_(
                ProposedEntity.source_artifact_id.in_(same_documents),
                and_(ProposedEntity.source_artifact_id.is_(None), ProposedEntity.run_id.in_(same_folder)),
            ),
        )
        .delete(synchronize_session=False)
    )

    reviewed: set[tuple[str, str]] = set()
    for row in session.query(ProposedEntity).filter(ProposedEntity.review_status != proposed):
        reviewed.add(_key(row.entity_type, row.title))
        reviewed.update(_key(row.entity_type, t) for t in (row.payload or {}).get(PROPOSED_AS, []))

    new = [r for r in records if _key(r.entity_type, r.title) not in reviewed]
    session.add_all(new)
    return {"added": len(new), "replaced": replaced, "already_reviewed": len(records) - len(new)}
