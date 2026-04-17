from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def invoice_output_path(invoice_number: str, base_dir: Path | None = None) -> Path:
    root = base_dir or Path.cwd()
    artifacts_dir = root / os.getenv("ARTIFACTS_DIR", "artifacts/invoices")
    ensure_dir(artifacts_dir)
    return artifacts_dir / f"{invoice_number}_{_timestamp()}.docx"


def metadata_output_path(docx_path: Path) -> Path:
    return docx_path.with_suffix(".metadata.json")


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

