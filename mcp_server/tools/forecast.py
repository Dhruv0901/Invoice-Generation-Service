from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp_server.services.audit import record_audit
from mcp_server.services.feature_builder import build_features
from mcp_server.services.model_loader import load_model_bundle


def forecast_sales(request: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    bundle = load_model_bundle(base_dir)
    features = build_features(request, bundle.manifest)
    expected_feature_count = getattr(bundle.model, "n_features_in_", None)
    if expected_feature_count is not None and len(features.ordered_values) != expected_feature_count:
        raise ValueError(
            "Feature manifest does not match model.pkl: "
            f"built {len(features.ordered_values)} features but model expects {expected_feature_count}."
        )
    prediction = bundle.model.predict([features.ordered_values])[0]

    result = {
        "request_id": request["request_id"],
        "target_month": request["target_month"],
        "product_id": request["product_id"],
        "predicted_monthly_sales": round(float(prediction), 4),
        "target_variable": bundle.manifest.get("target_variable", "monthly_sales"),
        "feature_count": len(features.ordered_names),
        "model_trace": {
            "model_type": bundle.manifest.get("model_type"),
            "model_training_timestamp": bundle.manifest.get("model_training_timestamp"),
        },
        "features": dict(zip(features.ordered_names, features.ordered_values)),
    }
    record_audit(
        {
            "event": "forecast_sales",
            "request_id": request["request_id"],
            "product_id": request["product_id"],
            "predicted_monthly_sales": result["predicted_monthly_sales"],
        },
        base_dir=base_dir,
    )
    return result
