from __future__ import annotations

EXTRACTION_SYSTEM_EN = """You are a consulting knowledge extraction assistant.
Extract structured master-data candidates from the provided engagement document chunks.
Return ONLY valid JSON matching the schema. Prefer precision over recall.
Language: respond field values in the document language (English or German); keep JSON keys in English.
"""

EXTRACTION_SYSTEM_DE = """Du bist ein Assistent zur Extraktion von Beratungswissen.
Extrahiere strukturierte Stammdaten-Kandidaten aus den gegebenen Engagement-Dokumentausschnitten.
Gib NUR gültiges JSON gemäß Schema zurück. Präzision vor Vollständigkeit.
Sprache: Feldwerte in der Dokumentsprache (Deutsch oder Englisch); JSON-Schlüssel auf Englisch belassen.
"""

EXTRACTION_SCHEMA_HINT = """
JSON schema:
{
  "clients": [{"name": str, "aliases": [str], "industry": str|null, "region": str|null, "status": str|null, "notes": str|null, "confidence": float, "chunk": int, "language": "en"|"de"|"mixed"}],
  "engagements": [{"title": str, "client_name": str|null, "type": str|null, "phase": str|null, "start": str|null, "end": str|null, "status": str|null, "commercial_model": str|null, "confidence": float, "chunk": int, "language": "en"|"de"|"mixed"}],
  "people": [{"name": str, "role": str|null, "org": str|null, "email": str|null, "engagement_titles": [str], "confidence": float, "chunk": int, "language": "en"|"de"|"mixed"}],
  "requirements": [{"statement": str, "priority": str|null, "status": str|null, "confidence": float, "chunk": int, "language": "en"|"de"|"mixed"}],
  "findings": [{"statement": str, "severity": str|null, "theme": str|null, "status": str|null, "confidence": float, "chunk": int, "language": "en"|"de"|"mixed"}],
  "deliverables": [{"name": str, "type": str|null, "due_date": str|null, "engagement_title": str|null, "confidence": float, "chunk": int, "language": "en"|"de"|"mixed"}]
}
"chunk" is the number in [brackets] of the excerpt the record was read from.
"""


def build_extraction_prompt(
    *,
    chunk_texts: list[str],
    theme_labels: list[str],
    prefer_de: bool = False,
) -> tuple[str, str]:
    system = EXTRACTION_SYSTEM_DE if prefer_de else EXTRACTION_SYSTEM_EN
    themes = ", ".join(theme_labels) if theme_labels else "(none yet)"
    # Cap each chunk to keep local CPU LLMs responsive (the caller decides how many chunks per call)
    capped = []
    for number, t in enumerate(chunk_texts, start=1):
        capped.append(f"[{number}]\n" + t[:900] + ("…" if len(t) > 900 else ""))
    joined = "\n\n---\n\n".join(capped)
    user = (
        f"Discovered themes / Gefundene Themen: {themes}\n\n"
        f"{EXTRACTION_SCHEMA_HINT}\n\n"
        f"Document chunks / Dokumentausschnitte:\n{joined}\n\n"
        "Extract entities now. JSON only."
    )
    return system, user
