from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelBundle:
    model: Any
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    model_path: Path
    feature_manifest_path: Path
    metrics_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_model_bundle(base_dir: Path | None = None) -> ModelBundle:
    base = base_dir or Path.cwd()
    forecast_core_dir = os.getenv("FORECAST_CORE_DIR", "forecast_core").strip()
    core_root = base / forecast_core_dir if forecast_core_dir else base

    configured_model_path = os.getenv("MODEL_PATH", "").strip()
    configured_manifest_path = os.getenv("FEATURE_MANIFEST_PATH", "").strip()
    configured_metrics_path = os.getenv("METRICS_PATH", "").strip()

    if configured_model_path:
        model_path = base / configured_model_path
    else:
        model_candidates = [core_root / "models/model.pkl", base / "models/model.pkl"]
        model_path = _first_existing_path(model_candidates) or (core_root / "models/model.pkl")

    if configured_manifest_path:
        manifest_path = base / configured_manifest_path
    else:
        manifest_candidates = [core_root / "models/feature_manifest.json", base / "models/feature_manifest.json"]
        manifest_path = _first_existing_path(manifest_candidates)
    if manifest_path is None:
        raise FileNotFoundError("No feature manifest found in forecast_core/models or models")

    if configured_metrics_path:
        metrics_path = base / configured_metrics_path
    else:
        metrics_candidates = [core_root / "reports/metrics.json", base / "reports/metrics.json"]
        metrics_path = _first_existing_path(metrics_candidates) or (core_root / "reports/metrics.json")

    manifest = _load_json(manifest_path)
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing at {model_path}")

    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    return ModelBundle(
        model=model,
        manifest=manifest,
        metrics=metrics,
        model_path=model_path,
        feature_manifest_path=manifest_path,
        metrics_path=metrics_path,
    )
