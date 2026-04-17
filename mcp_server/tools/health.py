from __future__ import annotations

from pathlib import Path

from mcp_server.services.model_loader import load_model_bundle
from mcp_server.services.qwen_invoice_renderer import _load_qwen_config


def health_check(base_dir: Path | None = None) -> dict[str, bool | str]:
    root = base_dir or Path.cwd()
    bundle = load_model_bundle(root)
    qwen_config = _load_qwen_config(root)
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_manifest_loaded": bundle.feature_manifest_path.exists(),
        "qwen_configured": bool(qwen_config["base_url"]),
        "qwen_model": qwen_config["model"],
        "heuristic_model": bundle.heuristic_fallback,
    }
