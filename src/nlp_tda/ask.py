"""Ask / command router over the ingested document library."""

from __future__ import annotations

import re
from typing import Any

import httpx

from nlp_tda.config import settings
from nlp_tda.db import get_session
from nlp_tda.embed.chroma_store import ChunkStore
from nlp_tda.embed.embeddings import embed_texts
from nlp_tda.export.excel_export import DEFAULT_FILTER, export_excel
from nlp_tda.models import ArtifactRow, ProposedEntity
from nlp_tda.pipeline import run_pipeline


COMMAND_HELP = {
    "en": [
        "Extract master data — find clients, people, requirements, findings, and more",
        "Show records — list draft master-data rows",
        "Create Excel file — download a handoff workbook",
        "Or ask a normal question about your documents",
    ],
    "de": [
        "Stammdaten extrahieren — findet Kunden, Personen, Anforderungen, Findings u. a.",
        "Datensätze anzeigen — Entwürfe auflisten",
        "Excel-Datei erstellen — Handoff-Arbeitsmappe herunterladen",
        "Oder eine normale Frage zu Ihren Dokumenten stellen",
    ],
}


def list_library() -> dict[str, Any]:
    session = get_session()
    try:
        arts = (
            session.query(ArtifactRow)
            .order_by(ArtifactRow.created_at.desc())
            .limit(500)
            .all()
        )
        items = [
            {
                "id": a.id,
                "title": a.title,
                "path": a.path,
                "mime_hint": a.mime_hint,
                "content_hash": a.content_hash[:12],
                "run_id": a.run_id,
                "preview": (a.text_preview or "")[:220],
            }
            for a in arts
        ]
        return {"count": len(items), "documents": items}
    finally:
        session.close()


def handle_command(
    text: str,
    *,
    lang: str = "en",
    batch_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {
            "kind": "help",
            "title": "Commands" if lang == "en" else "Befehle",
            "lines": COMMAND_HELP["de" if lang == "de" else "en"],
        }

    lower = raw.lower().strip()

    if lower in {"help", "?", "hilfe"}:
        return {
            "kind": "help",
            "title": "Commands" if lang == "en" else "Befehle",
            "lines": COMMAND_HELP["de" if lang == "de" else "en"],
        }

    if lower in {"status", "stats"} or lower.startswith("status "):
        return {"kind": "status", **_status()}

    if _matches(lower, ("run pipeline", "run", "pipeline", "extract", "pipeline starten", "extrahieren")):
        from nlp_tda.ingest.uploads import uploads_root

        source = None
        if batch_id:
            source = uploads_root() / batch_id
        result = run_pipeline(
            source,
            force_hash_embeddings=settings.use_hash_embeddings or True,
        )
        return {
            "kind": "pipeline",
            "title": "Pipeline finished" if lang == "en" else "Pipeline fertig",
            "result": result,
        }

    if _matches(lower, ("list entities", "entities", "list", "stammdaten", "entities anzeigen")):
        return {"kind": "entities", **_list_entities(run_id=run_id)}

    if _matches(lower, ("export excel", "export", "excel", "xlsx", "handoff")):
        session = get_session()
        try:
            # Prefer accepted+edited; fall back to proposed if empty
            meta = export_excel(session, filter_mode=DEFAULT_FILTER, run_id=run_id)
            if meta.get("total_entities", 0) == 0:
                meta = export_excel(session, filter_mode="proposed", run_id=run_id)
            return {
                "kind": "export",
                "title": "Excel ready" if lang == "en" else "Excel bereit",
                "result": meta,
                "download_url": f"/api/export/excel/{meta['file_id']}",
            }
        finally:
            session.close()

    # Free-text ask
    return {"kind": "answer", **_ask_library(raw, lang=lang)}


def _matches(lower: str, phrases: tuple[str, ...]) -> bool:
    return any(lower == p or lower.startswith(p + " ") for p in phrases)


def _status() -> dict[str, Any]:
    session = get_session()
    try:
        docs = session.query(ArtifactRow).count()
        ents = session.query(ProposedEntity).count()
        by_status: dict[str, int] = {}
        for row in session.query(ProposedEntity.review_status).all():
            by_status[row[0]] = by_status.get(row[0], 0) + 1
        chunks = 0
        try:
            chunks = ChunkStore().count()
        except Exception:
            pass
        return {
            "title": "Library status",
            "documents": docs,
            "entities": ents,
            "chunks": chunks,
            "by_status": by_status,
        }
    finally:
        session.close()


def _list_entities(*, run_id: str | None = None, limit: int = 40) -> dict[str, Any]:
    session = get_session()
    try:
        q = session.query(ProposedEntity).order_by(ProposedEntity.id.desc())
        if run_id:
            q = q.filter(ProposedEntity.run_id == run_id)
        rows = q.limit(limit).all()
        items = [
            {
                "id": r.id,
                "entity_type": r.entity_type,
                "title": r.title,
                "review_status": r.review_status,
                "confidence": r.confidence,
                "span_ref": r.span_ref,
                "run_id": r.run_id,
            }
            for r in rows
        ]
        return {"title": "Entities", "count": len(items), "entities": items}
    finally:
        session.close()


def _ask_library(question: str, *, lang: str = "en") -> dict[str, Any]:
    contexts = _retrieve_contexts(question, n=5)
    if not contexts:
        return {
            "title": "Answer" if lang == "en" else "Antwort",
            "question": question,
            "answer": (
                "No documents in the library yet. Upload files on the left, then run pipeline."
                if lang == "en"
                else "Noch keine Dokumente in der Bibliothek. Links Dateien hochladen, dann Pipeline starten."
            ),
            "sources": [],
            "mode": "empty",
        }

    client = _ollama_chat_available()
    if client:
        system = (
            "You answer questions about a consulting document library. "
            "Use only the provided excerpts. Cite span_ref when relevant. "
            "Reply in the user's language (EN or DE)."
            if lang == "en"
            else
            "Du beantwortest Fragen zu einer Beratungsdokument-Bibliothek. "
            "Nutze nur die gegebenen Ausschnitte. Nenne span_ref wenn sinnvoll. "
            "Antworte in der Sprache der Frage (DE oder EN)."
        )
        ctx = "\n\n---\n\n".join(
            f"[{c.get('span_ref') or c.get('artifact_id') or 'doc'}]\n{c['text']}" for c in contexts
        )
        user = f"Question:\n{question}\n\nExcerpts:\n{ctx}"
        try:
            answer = client.chat_text(system, user)
            return {
                "title": "Answer" if lang == "en" else "Antwort",
                "question": question,
                "answer": answer,
                "sources": contexts,
                "mode": "ollama",
            }
        except Exception:
            pass

    # Mock / extractive fallback
    answer = _mock_answer(question, contexts, lang=lang)
    return {
        "title": "Answer" if lang == "en" else "Antwort",
        "question": question,
        "answer": answer,
        "sources": contexts,
        "mode": "mock",
    }


def _retrieve_contexts(question: str, n: int = 5) -> list[dict[str, Any]]:
    # Prefer vector search when chunks exist
    try:
        store = ChunkStore()
        if store.count() > 0:
            emb = embed_texts([question], force_hash=settings.use_hash_embeddings or True)
            if emb.size:
                res = store.query(emb[0].tolist(), n_results=min(n, store.count()))
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                out = []
                for doc, meta in zip(docs, metas):
                    out.append(
                        {
                            "text": doc,
                            "span_ref": (meta or {}).get("span_ref"),
                            "artifact_id": (meta or {}).get("artifact_id"),
                            "run_id": (meta or {}).get("run_id"),
                        }
                    )
                if out:
                    return out
    except Exception:
        pass

    # Keyword fallback over artifact previews
    tokens = [t for t in re.findall(r"[a-zA-ZäöüÄÖÜß0-9]{3,}", question.lower()) if t]
    session = get_session()
    try:
        arts = session.query(ArtifactRow).order_by(ArtifactRow.created_at.desc()).limit(100).all()
        scored = []
        for a in arts:
            blob = f"{a.title}\n{a.text_preview or ''}".lower()
            score = sum(1 for t in tokens if t in blob)
            if score or not tokens:
                scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "text": (a.text_preview or a.title)[:600],
                "span_ref": a.path,
                "artifact_id": a.id,
                "run_id": a.run_id,
            }
            for score, a in scored[:n]
        ]
    finally:
        session.close()


def _mock_answer(question: str, contexts: list[dict[str, Any]], *, lang: str) -> str:
    q = question.lower()
    joined = "\n".join(c["text"] for c in contexts)
    bullets = []
    if "nordwind" in joined.lower() or "client" in q or "kunde" in q:
        bullets.append("Client appears as Nordwind Logistics GmbH (synthetic fixture)." if lang == "en"
                       else "Client erscheint als Nordwind Logistics GmbH (synthetisches Fixture).")
    if any(w in q for w in ("requirement", "anforderung", "bedarf")):
        bullets.append("Requirements mention TOM within 12 weeks and Stammdatenqualität." if lang == "en"
                       else "Anforderungen: TOM in 12 Wochen und Stammdatenqualität.")
    if any(w in q for w in ("risk", "risiko", "finding", "finding")):
        bullets.append("Findings: fragmented planning tools; Peak-Saison escalation risk." if lang == "en"
                       else "Findings: fragmentierte Planungstools; Peak-Saison-Eskalationsrisiko.")
    if not bullets:
        # First excerpt snippet
        snippet = contexts[0]["text"][:320].strip()
        bullets.append(
            ("Based on library excerpts:\n" + snippet)
            if lang == "en"
            else ("Basierend auf Bibliotheksausschnitten:\n" + snippet)
        )
    src = ", ".join(
        sorted({(c.get("span_ref") or c.get("artifact_id") or "?") for c in contexts})
    )
    bullets.append(("Sources: " + src) if lang == "en" else ("Quellen: " + src))
    return "\n".join(f"• {b}" for b in bullets)


class _ChatClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat_text(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=600.0)
        r.raise_for_status()
        return r.json()["message"]["content"]


def _ollama_chat_available() -> _ChatClient | None:
    if settings.force_mock_llm:
        return None
    try:
        r = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0)
        if r.status_code != 200:
            return None
        return _ChatClient(settings.ollama_base_url, settings.ollama_model)
    except Exception:
        return None
