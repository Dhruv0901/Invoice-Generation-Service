from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_audit(payload: dict[str, Any], base_dir: Path | None = None) -> None:
    root = base_dir or Path.cwd()
    audit_path = root / os.getenv("AUDIT_LOG_PATH", "artifacts/audit/audit.jsonl")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope) + "\n")

