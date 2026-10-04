from __future__ import annotations

import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nlp_tda.config import settings
from nlp_tda.ingest.parsers import (
    UPLOAD_SUFFIXES,
    expand_zip,
    parse_file,
    probe_filename,
)


def uploads_root() -> Path:
    root = Path(settings.uploads_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_batch() -> Path:
    batch_id = uuid.uuid4().hex[:12]
    path = uploads_root() / batch_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload(batch_dir: Path, filename: str, data: bytes) -> dict[str, Any]:
    """Save one uploaded file; expand zips; probe/parse each resulting file."""
    probe = probe_filename(filename)
    safe_name = Path(filename).name.replace("/", "_").replace("\\", "_")
    if not safe_name:
        return {
            "name": filename,
            "status": "error",
            "detected_type": "unknown",
            "message": "Empty filename",
        }

    if probe.status == "unsupported" or probe.status == "later":
        # Still store for transparency, but mark status
        dest = batch_dir / safe_name
        dest.write_bytes(data)
        return {
            "name": safe_name,
            "path": str(dest),
            "status": probe.status,
            "detected_type": probe.detected_type,
            "message": probe.message,
            "chars": 0,
        }

    if probe.status == "archive":
        zip_path = batch_dir / safe_name
        zip_path.write_bytes(data)
        expand_dir = batch_dir / f"_unzipped_{zip_path.stem}"
        try:
            extracted = expand_zip(zip_path, expand_dir)
        except Exception as exc:
            return {
                "name": safe_name,
                "path": str(zip_path),
                "status": "error",
                "detected_type": "zip",
                "message": f"Zip expand failed: {exc}",
                "chars": 0,
                "children": [],
            }
        children = []
        for child in extracted:
            children.append(_parse_saved(child))
        return {
            "name": safe_name,
            "path": str(zip_path),
            "status": "expanded",
            "detected_type": "zip",
            "message": f"Expanded {len(children)} file(s)",
            "chars": 0,
            "children": children,
        }

    # supported
    dest = batch_dir / safe_name
    # collision-safe
    if dest.exists():
        dest = batch_dir / f"{dest.stem}_{uuid.uuid4().hex[:6]}{dest.suffix}"
    dest.write_bytes(data)
    return _parse_saved(dest)


def _parse_saved(path: Path) -> dict[str, Any]:
    probe = probe_filename(path.name)
    if probe.status != "supported":
        return {
            "name": path.name,
            "path": str(path),
            "status": probe.status,
            "detected_type": probe.detected_type,
            "message": probe.message,
            "chars": 0,
        }
    try:
        doc = parse_file(path)
        status = "parsed"
        message = doc.warning or "OK"
        if doc.warning and not doc.text.strip():
            status = "warning"
        elif doc.warning:
            status = "parsed_with_warning"
        return {
            "name": path.name,
            "path": str(path),
            "status": status,
            "detected_type": doc.mime_hint,
            "message": message,
            "chars": len(doc.text),
            "title": doc.title,
        }
    except Exception as exc:
        return {
            "name": path.name,
            "path": str(path),
            "status": "error",
            "detected_type": probe.detected_type,
            "message": str(exc),
            "chars": 0,
        }


def copy_fixtures_into_batch(batch_dir: Path, fixtures: Path | None = None) -> list[dict[str, Any]]:
    """Helper for smoke tests / demo button."""
    src = fixtures or settings.fixtures_dir
    results = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in UPLOAD_SUFFIXES and path.suffix.lower() not in {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".csv",
            ".txt",
            ".md",
            ".eml",
            ".msg",
            ".zip",
        }:
            continue
        results.append(save_upload(batch_dir, path.name, path.read_bytes()))
    return results


def batch_summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    flat: list[dict[str, Any]] = []
    for f in files:
        if f.get("children"):
            flat.extend(f["children"])
        else:
            flat.append(f)
    ok = [f for f in flat if f.get("status") in {"parsed", "parsed_with_warning"}]
    return {
        "file_count": len(flat),
        "parsed_ok": len(ok),
        "errors": sum(1 for f in flat if f.get("status") == "error"),
        "warnings": sum(1 for f in flat if f.get("status") in {"warning", "parsed_with_warning", "later"}),
        "unsupported": sum(1 for f in flat if f.get("status") == "unsupported"),
        "files": files,
        "probe_schema": asdict(probe_filename("x.pdf")),
    }


def clear_batch(batch_id: str) -> None:
    path = uploads_root() / batch_id
    if path.exists():
        shutil.rmtree(path)
