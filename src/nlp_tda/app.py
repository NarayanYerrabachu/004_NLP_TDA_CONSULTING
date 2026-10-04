from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field

from nlp_tda.ask import handle_command, list_library
from nlp_tda.config import settings
from nlp_tda.db import get_engine, get_session
from nlp_tda.export.excel_export import (
    DEFAULT_FILTER,
    FILTER_MODES,
    count_by_entity,
    export_excel,
    find_export,
)
from nlp_tda.ingest.parsers import UPLOAD_SUFFIXES, probe_filename
from nlp_tda.ingest.uploads import batch_summary, create_batch, save_upload, uploads_root
from nlp_tda.models import ProposedEntity, ReviewStatus, ReviewUpdate
from nlp_tda.pipeline import run_pipeline

app = FastAPI(title="NLP_TDA", version="0.1.0")
templates = Jinja2Templates(directory=str(settings.templates_dir))


class AskRequest(BaseModel):
    text: str = Field(default="")
    lang: str = Field(default="en")
    batch_id: Optional[str] = None
    run_id: Optional[str] = None


WORKSPACE_STRINGS = {
    "en": {
        "brand": "Consulting Desk",
        "title": "Consulting Desk",
        "subtitle": "Turn engagement documents into clear master data you can review and export.",
        "nav_label": "Secondary pages",
        "lang_toggle": "Deutsch",
        "nav_review": "Review records",
        "nav_export": "Excel export",
        "steps_label": "How it works",
        "step1": "Add documents",
        "step2": "Extract or ask",
        "step3": "Use the results",
        "left_kicker": "Step 1",
        "left_title": "Your documents",
        "left_sub": "Add proposals, decks, emails, and sheets. Everything stays on this machine.",
        "drop_title": "Drop files here",
        "drop_hint": "PDF, Word, PowerPoint, Excel, CSV, text, email, or a ZIP folder.",
        "pick": "Browse files",
        "demo": "Try sample engagement",
        "refresh": "Refresh list",
        "docs_word": "documents",
        "lib_empty": "No documents yet. Drop a pack on the left to get started.",
        "mid_kicker": "Step 2",
        "mid_title": "Ask & extract",
        "mid_sub": "Create structured records from your documents, or ask a plain-language question.",
        "cmd_run": "Extract master data",
        "cmd_run_hint": "Find clients, people, requirements, findings, and more.",
        "cmd_list": "Show records",
        "cmd_list_hint": "Browse what was found so far.",
        "cmd_export": "Create Excel file",
        "cmd_export_hint": "Download a workbook for handoff.",
        "ask_placeholder": "e.g. Who is the client sponsor? What are the top risks?",
        "send": "Ask",
        "right_kicker": "Step 3",
        "right_title": "Results",
        "right_sub": "Answers, extracted records, and downloads appear here.",
        "out_empty": "When you extract data or ask a question, the results show up in this column.",
        "uploading": "Uploading your files…",
        "uploaded": "Files added",
        "uploaded_fmt": "Added {ok} of {total} files.",
        "uploaded_next": "Next: extract master data in the middle column.",
        "uploaded_chat": "Files are in your library. Click “Extract master data” when you’re ready.",
        "thinking": "One moment…",
        "you": "You",
        "assistant": "Assistant",
        "download": "Download Excel",
        "open_review": "Review & approve",
        "from_run": "from analysis",
        "status_ok": "Ready",
        "status_warn": "Check needed",
        "status_zip": "Unpacked",
        "status_error": "Couldn’t read",
        "status_later": "Not supported yet",
        "status_unsupported": "Unsupported",
        "status_fmt": "Library: {docs} documents · {ents} records · {chunks} text sections indexed.",
        "pipeline_title": "Extraction complete",
        "pipeline_lede": "Structured records are ready to review or export.",
        "pipeline_fmt": "Processed {docs} documents and drafted {ents} records across {themes} themes ({mode}).",
        "entities_title": "Draft records",
        "entities_fmt": "Found {n} draft records.",
        "export_title": "Excel workbook ready",
        "export_fmt": "Workbook includes {ents} records and {arts} source documents.",
        "export_ready": "Download the file, or open Review to accept or edit rows first.",
        "answer_title": "Answer",
        "none_yet": "Nothing to show yet.",
        "type_client": "Client",
        "type_engagement": "Engagement",
        "type_person": "Person",
        "type_requirement": "Requirement",
        "type_finding": "Finding",
        "type_deliverable": "Deliverable",
        "type_theme": "Theme",
        "type_artifact": "Document",
    },
    "de": {
        "brand": "Consulting Desk",
        "title": "Consulting Desk",
        "subtitle": "Aus Engagement-Dokumenten klare Stammdaten machen — prüfen und exportieren.",
        "nav_label": "Weitere Seiten",
        "lang_toggle": "English",
        "nav_review": "Datensätze prüfen",
        "nav_export": "Excel-Export",
        "steps_label": "So funktioniert’s",
        "step1": "Dokumente hinzufügen",
        "step2": "Extrahieren oder fragen",
        "step3": "Ergebnisse nutzen",
        "left_kicker": "Schritt 1",
        "left_title": "Ihre Dokumente",
        "left_sub": "Angebote, Decks, E-Mails und Tabellen hinzufügen. Alles bleibt auf diesem Rechner.",
        "drop_title": "Dateien hierher ziehen",
        "drop_hint": "PDF, Word, PowerPoint, Excel, CSV, Text, E-Mail oder ein ZIP-Ordner.",
        "pick": "Dateien wählen",
        "demo": "Beispiel-Engagement testen",
        "refresh": "Liste aktualisieren",
        "docs_word": "Dokumente",
        "lib_empty": "Noch keine Dokumente. Legen Sie links ein Paket ab, um zu starten.",
        "mid_kicker": "Schritt 2",
        "mid_title": "Fragen & extrahieren",
        "mid_sub": "Strukturierte Datensätze erzeugen oder eine Frage in Alltagssprache stellen.",
        "cmd_run": "Stammdaten extrahieren",
        "cmd_run_hint": "Findet Kunden, Personen, Anforderungen, Findings und mehr.",
        "cmd_list": "Datensätze anzeigen",
        "cmd_list_hint": "Bisher gefundene Entwürfe durchsehen.",
        "cmd_export": "Excel-Datei erstellen",
        "cmd_export_hint": "Arbeitsmappe für den Handoff herunterladen.",
        "ask_placeholder": "z. B. Wer ist der Auftraggeber? Was sind die größten Risiken?",
        "send": "Fragen",
        "right_kicker": "Schritt 3",
        "right_title": "Ergebnisse",
        "right_sub": "Antworten, extrahierte Datensätze und Downloads erscheinen hier.",
        "out_empty": "Wenn Sie Daten extrahieren oder eine Frage stellen, erscheinen die Ergebnisse in dieser Spalte.",
        "uploading": "Dateien werden hochgeladen…",
        "uploaded": "Dateien hinzugefügt",
        "uploaded_fmt": "{ok} von {total} Dateien hinzugefügt.",
        "uploaded_next": "Als Nächstes: in der mittleren Spalte Stammdaten extrahieren.",
        "uploaded_chat": "Die Dateien sind in Ihrer Bibliothek. Klicken Sie auf „Stammdaten extrahieren“, wenn Sie bereit sind.",
        "thinking": "Einen Moment…",
        "you": "Sie",
        "assistant": "Assistent",
        "download": "Excel herunterladen",
        "open_review": "Prüfen & freigeben",
        "from_run": "aus Analyse",
        "status_ok": "Bereit",
        "status_warn": "Bitte prüfen",
        "status_zip": "Entpackt",
        "status_error": "Nicht lesbar",
        "status_later": "Noch nicht unterstützt",
        "status_unsupported": "Nicht unterstützt",
        "status_fmt": "Bibliothek: {docs} Dokumente · {ents} Datensätze · {chunks} Textabschnitte indexiert.",
        "pipeline_title": "Extraktion abgeschlossen",
        "pipeline_lede": "Strukturierte Datensätze sind bereit zur Prüfung oder zum Export.",
        "pipeline_fmt": "{docs} Dokumente verarbeitet und {ents} Datensätze in {themes} Themen entworfen ({mode}).",
        "entities_title": "Entwürfe",
        "entities_fmt": "{n} Entwurfs-Datensätze gefunden.",
        "export_title": "Excel-Arbeitsmappe bereit",
        "export_fmt": "Arbeitsmappe enthält {ents} Datensätze und {arts} Quelldokumente.",
        "export_ready": "Datei herunterladen — oder zuerst unter Prüfung Zeilen annehmen/bearbeiten.",
        "answer_title": "Antwort",
        "none_yet": "Noch nichts anzuzeigen.",
        "type_client": "Kunde",
        "type_engagement": "Engagement",
        "type_person": "Person",
        "type_requirement": "Anforderung",
        "type_finding": "Finding",
        "type_deliverable": "Deliverable",
        "type_theme": "Thema",
        "type_artifact": "Dokument",
    },
}

REVIEW_STRINGS = {
    "en": {
        "title": "Review records",
        "subtitle": "Check what was extracted. Accept good rows, edit details, or reject mistakes.",
        "run": "Try sample engagement",
        "filter_all": "All",
        "proposed": "Needs review",
        "accepted": "Accepted",
        "rejected": "Rejected",
        "edited": "Edited",
        "accept": "Accept",
        "reject": "Reject",
        "save": "Save edits",
        "confidence": "Confidence",
        "source": "Source",
        "empty": "Nothing to review yet. Go home, add documents, then click Extract master data.",
        "details": "Show technical details",
        "details_hint": "Only needed if you want to tweak the raw fields.",
        "lang_toggle": "Deutsch",
        "synthetic_note": "Sample data in this repo is synthetic — not real client material.",
        "nav_ingest": "Back to desk",
        "nav_review": "Review",
        "nav_export": "Excel export",
        "goto_export": "Create Excel",
    },
    "de": {
        "title": "Datensätze prüfen",
        "subtitle": "Extraktion prüfen. Gute Zeilen annehmen, Details bearbeiten oder Fehler ablehnen.",
        "run": "Beispiel-Engagement testen",
        "filter_all": "Alle",
        "proposed": "Zu prüfen",
        "accepted": "Angenommen",
        "rejected": "Abgelehnt",
        "edited": "Bearbeitet",
        "accept": "Annehmen",
        "reject": "Ablehnen",
        "save": "Änderungen speichern",
        "confidence": "Sicherheit",
        "source": "Quelle",
        "empty": "Noch nichts zu prüfen. Zurück zum Desk, Dokumente hinzufügen, dann Stammdaten extrahieren.",
        "details": "Technische Details anzeigen",
        "details_hint": "Nur nötig, wenn Sie Rohfelder anpassen wollen.",
        "lang_toggle": "English",
        "synthetic_note": "Beispieldaten in diesem Repo sind synthetisch — keine echten Kundendaten.",
        "nav_ingest": "Zurück zum Desk",
        "nav_review": "Prüfung",
        "nav_export": "Excel-Export",
        "goto_export": "Excel erstellen",
    },
}

INGEST_STRINGS = {
    "en": {
        "title": "NLP_TDA Ingest",
        "subtitle": "Drop consulting packs — Office, PDF, email, sheets — then run extraction.",
        "lang_toggle": "DE",
        "nav_ingest": "Workspace",
        "nav_review": "Review",
        "nav_export": "Excel export",
        "drop_title": "Drag & drop files here",
        "drop_hint": "Or use the file picker. Multi-select supported. Zip archives are expanded.",
        "pick": "Choose files",
        "run": "Run pipeline on uploaded files",
        "running": "Running pipeline…",
        "clear": "Clear list",
        "demo": "Load synthetic fixtures",
        "supported": "Accepted now",
        "supported_list": "PDF (text), DOCX, PPTX, XLSX, CSV, TXT/MD, EML, MSG, ZIP",
        "later": "Later / not yet",
        "later_list": "Scan-only image PDFs (OCR), image files, .msg edge cases may vary",
        "col_name": "File",
        "col_type": "Type",
        "col_status": "Status",
        "col_msg": "Detail",
        "empty": "No files yet.",
        "done": "Pipeline finished",
        "goto_review": "Open review queue",
        "goto_export": "Export Excel",
        "note": "Files stay local under data/uploads/. Ollama mock + hash embeddings used when models are unavailable.",
        "error_none": "Upload at least one parseable file before running.",
    },
    "de": {
        "title": "NLP_TDA Import",
        "subtitle": "Beratungspakete ablegen — Office, PDF, E-Mail, Tabellen — dann Extraktion starten.",
        "lang_toggle": "EN",
        "nav_ingest": "Arbeitsfläche",
        "nav_review": "Prüfung",
        "nav_export": "Excel-Export",
        "drop_title": "Dateien hierher ziehen",
        "drop_hint": "Oder Dateiauswahl nutzen. Mehrfachauswahl möglich. ZIP-Archive werden entpackt.",
        "pick": "Dateien wählen",
        "run": "Pipeline auf Uploads starten",
        "running": "Pipeline läuft…",
        "clear": "Liste leeren",
        "demo": "Synthetische Fixtures laden",
        "supported": "Jetzt akzeptiert",
        "supported_list": "PDF (Text), DOCX, PPTX, XLSX, CSV, TXT/MD, EML, MSG, ZIP",
        "later": "Später / noch nicht",
        "later_list": "Reine Scan-PDFs (OCR), Bilddateien; .msg-Grenzfälle möglich",
        "col_name": "Datei",
        "col_type": "Typ",
        "col_status": "Status",
        "col_msg": "Detail",
        "empty": "Noch keine Dateien.",
        "done": "Pipeline fertig",
        "goto_review": "Prüfungsqueue öffnen",
        "goto_export": "Excel exportieren",
        "note": "Dateien bleiben lokal unter data/uploads/. Ollama-Mock + Hash-Embeddings wenn Modelle fehlen.",
        "error_none": "Mindestens eine parsebare Datei hochladen, bevor die Pipeline startet.",
    },
}

EXPORT_STRINGS = {
    "en": {
        "title": "Excel export",
        "subtitle": "Create a multi-sheet workbook for handoff — clients, people, requirements, findings, and more.",
        "note": "By default we include only accepted and edited rows. Switch the filter if you want drafts too.",
        "lang_toggle": "Deutsch",
        "nav_ingest": "Back to desk",
        "nav_review": "Review records",
        "nav_export": "Excel export",
        "config": "What to include",
        "filter_label": "Which records?",
        "filter_accepted_edited": "Accepted + edited (recommended)",
        "filter_accepted": "Accepted only",
        "filter_proposed": "Needs review only",
        "filter_all": "Everything (including rejected)",
        "filter_hint_accepted_edited": "Best for handoff after you’ve reviewed the drafts.",
        "filter_hint_accepted": "Only rows you marked as accepted.",
        "filter_hint_proposed": "Drafts that still need a human look.",
        "filter_hint_all": "Includes rejected rows — usually for auditing.",
        "run_label": "Optional analysis ID",
        "run_placeholder": "Leave blank to include all matching analyses",
        "preview": "Refresh preview",
        "generate": "Create Excel file",
        "counts_title": "What you’ll get",
        "counts_empty": "Choose a filter to see counts.",
        "ready": "Your workbook is ready",
        "download": "Download Excel",
        "entity_client": "Client",
        "entity_engagement": "Engagement",
        "entity_person": "Person",
        "entity_artifact": "Document",
        "entity_requirement": "Requirement",
        "entity_finding": "Finding",
        "entity_deliverable": "Deliverable",
        "entity_theme": "Theme",
    },
    "de": {
        "title": "Excel-Export",
        "subtitle": "Mehrblatt-Arbeitsmappe für den Handoff — Kunden, Personen, Anforderungen, Findings und mehr.",
        "note": "Standardmäßig nur angenommene und bearbeitete Zeilen. Filter wechseln, wenn Sie auch Entwürfe wollen.",
        "lang_toggle": "English",
        "nav_ingest": "Zurück zum Desk",
        "nav_review": "Datensätze prüfen",
        "nav_export": "Excel-Export",
        "config": "Was soll rein?",
        "filter_label": "Welche Datensätze?",
        "filter_accepted_edited": "Angenommen + bearbeitet (empfohlen)",
        "filter_accepted": "Nur angenommen",
        "filter_proposed": "Nur zu prüfen",
        "filter_all": "Alles (inkl. abgelehnt)",
        "filter_hint_accepted_edited": "Ideal für den Handoff nach der Prüfung.",
        "filter_hint_accepted": "Nur Zeilen, die Sie angenommen haben.",
        "filter_hint_proposed": "Entwürfe, die noch geprüft werden müssen.",
        "filter_hint_all": "Inklusive abgelehnt — eher für Audits.",
        "run_label": "Optionale Analyse-ID",
        "run_placeholder": "Leer = alle passenden Analysen",
        "preview": "Vorschau aktualisieren",
        "generate": "Excel-Datei erstellen",
        "counts_title": "Was Sie erhalten",
        "counts_empty": "Filter wählen, um Anzahlen zu sehen.",
        "ready": "Ihre Arbeitsmappe ist bereit",
        "download": "Excel herunterladen",
        "entity_client": "Kunde",
        "entity_engagement": "Engagement",
        "entity_person": "Person",
        "entity_artifact": "Dokument",
        "entity_requirement": "Anforderung",
        "entity_finding": "Finding",
        "entity_deliverable": "Deliverable",
        "entity_theme": "Thema",
    },
}


@app.on_event("startup")
def _startup() -> None:
    get_engine()
    settings.project_root.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Path(settings.exports_dir).mkdir(parents=True, exist_ok=True)
    uploads_root()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/formats")
def list_formats() -> dict:
    return {
        "accepted": sorted(s.lstrip(".") for s in UPLOAD_SUFFIXES),
        "pipeline": sorted(
            s.lstrip(".")
            for s in {
                ".pdf",
                ".docx",
                ".pptx",
                ".xlsx",
                ".csv",
                ".txt",
                ".md",
                ".eml",
                ".msg",
            }
        ),
        "later": ["png", "jpg", "jpeg", "tif", "tiff", "gif", "webp", "scan-pdf-ocr"],
    }


@app.post("/api/ingest/upload")
async def api_upload(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files")
    batch_dir = create_batch()
    results = []
    for uf in files:
        data = await uf.read()
        name = uf.filename or "upload.bin"
        results.append(save_upload(batch_dir, name, data))
    summary = batch_summary(results)
    return {
        "batch_id": batch_dir.name,
        "batch_dir": str(batch_dir),
        **summary,
    }


@app.post("/api/ingest/probe")
def api_probe(names: list[str]) -> list[dict]:
    return [
        {
            "name": p.name,
            "suffix": p.suffix,
            "detected_type": p.detected_type,
            "status": p.status,
            "message": p.message,
        }
        for p in (probe_filename(n) for n in names)
    ]


@app.post("/api/pipeline/run")
def api_run_pipeline(
    source_dir: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> dict:
    if batch_id:
        path = uploads_root() / batch_id
    elif source_dir:
        path = Path(source_dir)
    else:
        path = settings.fixtures_dir
    try:
        return run_pipeline(path, force_hash_embeddings=settings.use_hash_embeddings or None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/entities")
def list_entities(
    review_status: Optional[str] = None,
    entity_type: Optional[str] = None,
    run_id: Optional[str] = None,
) -> list[dict]:
    session = get_session()
    try:
        q = session.query(ProposedEntity)
        if review_status:
            q = q.filter(ProposedEntity.review_status == review_status)
        if entity_type:
            q = q.filter(ProposedEntity.entity_type == entity_type)
        if run_id:
            q = q.filter(ProposedEntity.run_id == run_id)
        rows = q.order_by(ProposedEntity.id.desc()).all()
        return [_entity_dict(r) for r in rows]
    finally:
        session.close()


@app.get("/api/entities/{entity_id}")
def get_entity(entity_id: int) -> dict:
    session = get_session()
    try:
        row = session.get(ProposedEntity, entity_id)
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return _entity_dict(row)
    finally:
        session.close()


@app.patch("/api/entities/{entity_id}")
def patch_entity(entity_id: int, update: ReviewUpdate) -> dict:
    session = get_session()
    try:
        row = session.get(ProposedEntity, entity_id)
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        if update.edits:
            payload = dict(row.payload or {})
            payload.update(update.edits)
            row.payload = payload
            if "name" in update.edits:
                row.title = str(update.edits["name"])
            elif "title" in update.edits:
                row.title = str(update.edits["title"])
            elif "statement" in update.edits:
                row.title = str(update.edits["statement"])[:120]
            if update.review_status == ReviewStatus.proposed:
                row.review_status = ReviewStatus.edited.value
            else:
                row.review_status = update.review_status.value
        else:
            row.review_status = update.review_status.value
        session.commit()
        session.refresh(row)
        return _entity_dict(row)
    finally:
        session.close()


@app.get("/api/export/filters")
def api_export_filters() -> dict:
    return {"filters": FILTER_MODES, "default": DEFAULT_FILTER}


@app.get("/api/export/preview")
def api_export_preview(
    filter_mode: str = Query(default=DEFAULT_FILTER),
    run_id: Optional[str] = None,
) -> dict:
    session = get_session()
    try:
        try:
            return count_by_entity(session, filter_mode=filter_mode, run_id=run_id or None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@app.post("/api/export/excel")
def api_export_excel(
    filter_mode: str = Query(default=DEFAULT_FILTER),
    run_id: Optional[str] = None,
) -> dict:
    session = get_session()
    try:
        try:
            return export_excel(session, filter_mode=filter_mode, run_id=run_id or None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@app.get("/api/export/excel/{file_id}")
def api_download_excel(file_id: str):
    path = find_export(file_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


@app.get("/api/library")
def api_library() -> dict:
    return list_library()


@app.post("/api/ask")
def api_ask(body: AskRequest) -> dict:
    try:
        return handle_command(
            body.text,
            lang="de" if body.lang == "de" else "en",
            batch_id=body.batch_id,
            run_id=body.run_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def workspace_ui(request: Request, lang: str = Query(default="en")):
    lang = "de" if lang == "de" else "en"
    return templates.TemplateResponse(
        request,
        "workspace.html",
        {
            "s": WORKSPACE_STRINGS[lang],
            "lang": lang,
            "other_lang": "en" if lang == "de" else "de",
        },
    )


@app.get("/ingest", response_class=HTMLResponse)
def ingest_ui(request: Request, lang: str = Query(default="en")):
    lang = "de" if lang == "de" else "en"
    return templates.TemplateResponse(
        request,
        "ingest.html",
        {
            "s": INGEST_STRINGS[lang],
            "lang": lang,
            "other_lang": "en" if lang == "de" else "de",
        },
    )


@app.get("/review", response_class=HTMLResponse)
def review_ui(
    request: Request,
    lang: str = Query(default="en"),
    status: str = Query(default="proposed"),
    run_id: Optional[str] = Query(default=None),
):
    lang = "de" if lang == "de" else "en"
    strings = REVIEW_STRINGS[lang]
    session = get_session()
    try:
        q = session.query(ProposedEntity)
        if status != "all":
            q = q.filter(ProposedEntity.review_status == status)
        if run_id:
            q = q.filter(ProposedEntity.run_id == run_id)
        rows = q.order_by(ProposedEntity.id.desc()).limit(200).all()
        entities = [_entity_dict(r) for r in rows]
    finally:
        session.close()
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "s": strings,
            "lang": lang,
            "other_lang": "en" if lang == "de" else "de",
            "status": status,
            "entities": entities,
            "run_id": run_id or "",
        },
    )


@app.get("/export", response_class=HTMLResponse)
def export_ui(
    request: Request,
    lang: str = Query(default="en"),
    run_id: Optional[str] = Query(default=None),
):
    lang = "de" if lang == "de" else "en"
    return templates.TemplateResponse(
        request,
        "export.html",
        {
            "s": EXPORT_STRINGS[lang],
            "lang": lang,
            "other_lang": "en" if lang == "de" else "de",
            "run_id": run_id or "",
        },
    )


@app.post("/ui/run")
def ui_run(lang: str = Form(default="en")):
    run_pipeline(settings.fixtures_dir)
    return RedirectResponse(url=f"/review?lang={lang}&status=proposed", status_code=303)


@app.post("/ui/entities/{entity_id}/review")
def ui_review(
    entity_id: int,
    action: str = Form(...),
    payload_json: str = Form(default=""),
    lang: str = Form(default="en"),
    status: str = Form(default="proposed"),
):
    session = get_session()
    try:
        row = session.get(ProposedEntity, entity_id)
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        if action == "accept":
            row.review_status = ReviewStatus.accepted.value
        elif action == "reject":
            row.review_status = ReviewStatus.rejected.value
        elif action == "edit":
            try:
                edits = json.loads(payload_json) if payload_json.strip() else {}
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON") from exc
            payload = dict(row.payload or {})
            payload.update(edits)
            row.payload = payload
            row.review_status = ReviewStatus.edited.value
            if "name" in edits:
                row.title = str(edits["name"])
            elif "title" in edits:
                row.title = str(edits["title"])
            elif "statement" in edits:
                row.title = str(edits["statement"])[:120]
        session.commit()
    finally:
        session.close()
    return RedirectResponse(url=f"/review?lang={lang}&status={status}", status_code=303)


def _entity_dict(row: ProposedEntity) -> dict:
    payload = row.payload or {}
    skip = {
        "confidence",
        "source_artifact_id",
        "span_ref",
        "review_status",
        "language",
        "run_id",
        "entity_type",
    }
    fields: list[dict[str, str]] = []
    for key, value in payload.items():
        if key in skip:
            continue
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        fields.append({"key": key.replace("_", " ").title(), "value": text})
    return {
        "id": row.id,
        "entity_type": row.entity_type,
        "title": row.title,
        "payload": payload,
        "confidence": row.confidence,
        "source_artifact_id": row.source_artifact_id,
        "span_ref": row.span_ref,
        "review_status": row.review_status,
        "language": row.language,
        "run_id": row.run_id,
        "payload_pretty": json.dumps(payload, ensure_ascii=False, indent=2),
        "fields": fields,
    }
