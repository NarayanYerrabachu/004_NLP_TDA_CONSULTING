from __future__ import annotations

import json
import re
from typing import Any

import httpx

from nlp_tda.config import settings
from nlp_tda.extract.dedupe import merge_near_duplicates
from nlp_tda.extract.prompts import build_extraction_prompt
from nlp_tda.ingest.chunking import Chunk
from nlp_tda.models import ExtractionBundle


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    def available(self) -> bool:
        if settings.force_mock_llm:
            return False
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 2048,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # CPU 7B on long consulting prompts can exceed 3 minutes
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=600.0)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return _parse_json_object(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    return json.loads(match.group(0))


def mock_extract(chunk_texts: list[str], theme_labels: list[str]) -> ExtractionBundle:
    """Deterministic fixture-aware extraction when Ollama is unavailable."""
    blob = "\n".join(chunk_texts).lower()
    prefer_de = any(w in blob for w in ("anforderungen", "mandat", "liefergegenstand", "risiko"))

    clients = []
    engagements = []
    people = []
    requirements = []
    findings = []
    deliverables = []

    if "nordwind" in blob or "acme" in blob or "synthetic" in blob:
        clients.append(
            {
                "name": "Nordwind Logistics GmbH" if prefer_de or "nordwind" in blob else "Nordwind Logistics GmbH",
                "aliases": ["Nordwind", "NW Logistics"],
                "industry": "Logistics / Logistik",
                "region": "DACH",
                "status": "active",
                "notes": "SYNTHETIC fixture client — not real IP",
                "confidence": 0.92,
                "span_ref": "proposal",
                "language": "de" if prefer_de else "mixed",
            }
        )
        engagements.append(
            {
                "title": "Operating Model Redesign 2026" if not prefer_de else "Neugestaltung Operating Model 2026",
                "client_name": "Nordwind Logistics GmbH",
                "type": "operations",
                "phase": "discovery",
                "start": "2026-01-15",
                "end": "2026-06-30",
                "status": "proposed",
                "commercial_model": "fixed fee + expenses",
                "confidence": 0.88,
                "span_ref": "sow",
                "language": "mixed",
            }
        )
        people.append(
            {
                "name": "Dr. Lena Hartmann",
                "role": "Client Sponsor / Auftraggeberin",
                "org": "Nordwind Logistics GmbH",
                "email": "lena.hartmann@example.invalid",
                "engagement_titles": ["Operating Model Redesign 2026"],
                "confidence": 0.85,
                "span_ref": "kickoff",
                "language": "de",
            }
        )
        people.append(
            {
                "name": "Marcus Lee",
                "role": "Engagement Manager",
                "org": "Harbor Advisory",
                "email": "marcus.lee@example.invalid",
                "engagement_titles": ["Operating Model Redesign 2026"],
                "confidence": 0.84,
                "span_ref": "kickoff",
                "language": "en",
            }
        )
        requirements.append(
            {
                "statement": "Define a target operating model for warehouse + transport planning within 12 weeks.",
                "priority": "high",
                "status": "open",
                "confidence": 0.8,
                "span_ref": "sow",
                "language": "en",
            }
        )
        requirements.append(
            {
                "statement": "Die Stammdatenqualität für Artikel und Standorte muss messbar verbessert werden.",
                "priority": "high",
                "status": "open",
                "confidence": 0.81,
                "span_ref": "anforderungen",
                "language": "de",
            }
        )
        findings.append(
            {
                "statement": "Planning tools are fragmented across three regional hubs with weak master-data ownership.",
                "severity": "medium",
                "theme": theme_labels[0] if theme_labels else "operations",
                "status": "open",
                "confidence": 0.78,
                "span_ref": "findings",
                "language": "en",
            }
        )
        findings.append(
            {
                "statement": "Fehlende End-to-End-Prozessverantwortung erhöht Eskalationsaufwand bei Peak-Saison.",
                "severity": "high",
                "theme": theme_labels[0] if theme_labels else "governance",
                "status": "open",
                "confidence": 0.79,
                "span_ref": "risiken",
                "language": "de",
            }
        )
        deliverables.append(
            {
                "name": "Target Operating Model blueprint",
                "type": "report",
                "due_date": "2026-04-30",
                "engagement_title": "Operating Model Redesign 2026",
                "confidence": 0.86,
                "span_ref": "sow",
                "language": "en",
            }
        )
        deliverables.append(
            {
                "name": "Roadmap Implementierung (90 Tage)",
                "type": "roadmap",
                "due_date": "2026-05-15",
                "engagement_title": "Operating Model Redesign 2026",
                "confidence": 0.83,
                "span_ref": "liefergegenstaende",
                "language": "de",
            }
        )
    else:
        # Generic minimal proposals from first chunk
        snippet = chunk_texts[0][:240] if chunk_texts else "untitled"
        clients.append(
            {
                "name": "Unknown Client",
                "aliases": [],
                "industry": None,
                "region": None,
                "status": "unknown",
                "notes": snippet,
                "confidence": 0.3,
                "span_ref": None,
                "language": "mixed",
            }
        )

    return ExtractionBundle.model_validate(
        {
            "clients": clients,
            "engagements": engagements,
            "people": people,
            "requirements": requirements,
            "findings": findings,
            "deliverables": deliverables,
        }
    )


def extract_entities(
    chunks: list[Chunk],
    theme_labels: list[str],
    *,
    prefer_de: bool = False,
) -> tuple[ExtractionBundle, str, int]:
    """Extract entities from every chunk of the pack, one LLM call per batch.

    Returns (bundle, llm_mode, chunks_read). llm_mode: "mock" (no LLM), "ollama" (every batch
    answered), "ollama-partial" (some batches failed and were skipped), "mock-fallback"
    (every batch failed). Records from the LLM point at the chunk they were read from, and
    requirements / findings stated twice in other words are merged.
    """
    chunk_texts = [c.text for c in chunks]
    client = OllamaClient()
    if not client.available():
        return mock_extract(chunk_texts, theme_labels), "mock", len(chunks)

    batches = _batches(chunks)
    merged = ExtractionBundle()
    seen: set[tuple[str, str]] = set()
    failed = 0
    for batch in batches:
        system, user = build_extraction_prompt(
            chunk_texts=[c.text for c in batch],
            theme_labels=theme_labels,
            prefer_de=prefer_de,
        )
        try:
            bundle = ExtractionBundle.model_validate(client.chat_json(system, user))
        except Exception:
            failed += 1
            continue
        _attach_sources(bundle, batch)
        _merge(merged, bundle, seen)

    chunks_read = sum(len(b) for b in batches)
    if batches and failed == len(batches):
        return mock_extract(chunk_texts, theme_labels), "mock-fallback", chunks_read
    merge_near_duplicates(merged)
    return merged, "ollama-partial" if failed else "ollama", chunks_read


# The field that names a record, per entity list: the same name twice is one record.
_NAME_FIELD = {
    "clients": "name",
    "engagements": "title",
    "people": "name",
    "requirements": "statement",
    "findings": "statement",
    "deliverables": "name",
}


def _batches(chunks: list[Chunk]) -> list[list[Chunk]]:
    size = max(settings.extract_batch_chunks, 1)
    batches = [chunks[i : i + size] for i in range(0, len(chunks), size)]
    limit = settings.extract_max_batches
    if limit > 0 and len(batches) > limit:
        step = len(batches) / limit
        batches = [batches[int(i * step)] for i in range(limit)]
    return batches


def _norm(text: object) -> str:
    return " ".join(str(text).lower().split())


def _attach_sources(bundle: ExtractionBundle, batch: list[Chunk]) -> None:
    """Point each record at the chunk it was read from.

    The excerpt that contains the record's name or statement word for word wins; otherwise the
    excerpt number the LLM cited. A record with neither has no source (never an invented one).
    """
    texts = [_norm(c.text) for c in batch]
    for field, name_field in _NAME_FIELD.items():
        for item in getattr(bundle, field):
            name = _norm(getattr(item, name_field))
            chunk = next((c for c, text in zip(batch, texts) if name and name in text), None)
            if chunk is None:
                digits = "".join(ch for ch in str(item.chunk) if ch.isdigit())
                number = int(digits) if digits else 0
                chunk = batch[number - 1] if 1 <= number <= len(batch) else None
            item.source_artifact_id = chunk.artifact_id if chunk else None
            item.span_ref = chunk.span_ref if chunk else None


def _merge(merged: ExtractionBundle, bundle: ExtractionBundle, seen: set[tuple[str, str]]) -> None:
    """Add a batch's records to ``merged``; a record already named by an earlier batch is kept once."""
    for field, name_field in _NAME_FIELD.items():
        for item in getattr(bundle, field):
            key = (field, _norm(getattr(item, name_field)))
            if key in seen:
                continue
            seen.add(key)
            getattr(merged, field).append(item)
