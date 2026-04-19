from __future__ import annotations

from pathlib import Path

from mcp_server.services.model_loader import load_model_bundle


def get_model_metadata(base_dir: Path | None = None) -> dict:
    bundle = load_model_bundle(base_dir)
    manifest = bundle.manifest
    return {
        "model_name": manifest.get("model_name", "future_sales_forecaster"),
        "target_variable": manifest.get("target_variable", "monthly_sales"),
        "feature_count": len(manifest.get("ordered_feature_names", [])),
        "training_split_strategy": manifest.get("data_split_strategy", "unknown"),
        "holdout_months": manifest.get("holdout_months"),
        "metrics": bundle.metrics,
    }
