from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp_server.services.audit import record_audit
from mcp_server.services.pricing import build_invoice_pricing
from mcp_server.services.qwen_invoice_renderer import render_invoice_docx
from mcp_server.services.storage import invoice_output_path, metadata_output_path, write_metadata
from mcp_server.tools.forecast import forecast_sales
from scripts.generate_next_month_forecast import build_invoice_requests, default_params


def _invoice_number(request_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"PFI-{stamp}-{request_id}"


def generate_invoice_docx(request: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    forecast_result = forecast_sales(request, base_dir=base_dir)
    tax_rate = float(request.get("tax_rate", 0.0))
    pricing = build_invoice_pricing(request, forecast_result["predicted_monthly_sales"], tax_rate)

    invoice_number = _invoice_number(request["request_id"])
    docx_path = invoice_output_path(invoice_number, base_dir=base_dir)
    invoice_payload = {
        "invoice_number": invoice_number,
        "request_id": request["request_id"],
        "target_month": request["target_month"],
        "product_id": request["product_id"],
        "product_name": request["product_context"]["product_name"],
        "category": request["product_context"]["category"],
        "billed_quantity": pricing.billed_quantity,
        "unit_price": pricing.unit_price,
        "line_total": pricing.line_total,
        "subtotal": pricing.subtotal,
        "tax": pricing.tax,
        "grand_total": pricing.grand_total,
    }
    render_details = render_invoice_docx(docx_path, invoice_payload)

    metadata = {
        **invoice_payload,
        "predicted_monthly_sales": forecast_result["predicted_monthly_sales"],
        "model_trace": forecast_result["model_trace"],
        "artifact_path": str(docx_path),
        **render_details,
    }
    write_metadata(metadata_output_path(docx_path), metadata)
    record_audit(
        {
            "event": "generate_invoice_docx",
            "request_id": request["request_id"],
            "invoice_number": invoice_number,
            "artifact_path": str(docx_path),
        },
        base_dir=base_dir,
    )

    return {
        "invoice_number": invoice_number,
        "artifact_path": str(docx_path),
        "metadata_path": str(metadata_output_path(docx_path)),
        "invoice": invoice_payload,
        "forecast": forecast_result,
        "render_details": render_details,
    }


def forecast_and_generate_invoice(request: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    return generate_invoice_docx(request, base_dir=base_dir)


def generate_invoices_from_kaggle_sample(request: dict[str, Any] | None = None, base_dir: Path | None = None) -> dict[str, Any]:
    resolved_base_dir = base_dir or Path.cwd()
    payload = request or {}
    params = default_params(payload.get("source_extract_dir"))

    dataset_cfg = params.setdefault("dataset", {})
    dataset_cfg["kaggle_dataset"] = payload.get(
        "kaggle_dataset",
        dataset_cfg.get("kaggle_dataset", "amirkhanh/synthetic-retail-dataset-1-2m-transactions"),
    )
    dataset_cfg["source_extract_dir"] = payload.get(
        "source_extract_dir",
        dataset_cfg["source_extract_dir"],
    )

    requests, next_month = build_invoice_requests(
        params=params,
        item_count=int(payload.get("sample_size", 30)),
        tax_rate=float(payload.get("tax_rate", 0.1)),
        random_seed=int(payload.get("random_seed", 42)),
    )

    invoices = [generate_invoice_docx(invoice_request, base_dir=resolved_base_dir) for invoice_request in requests]
    record_audit(
        {
            "event": "generate_invoices_from_kaggle_sample",
            "sample_size": len(invoices),
            "target_month": next_month,
            "kaggle_dataset": dataset_cfg["kaggle_dataset"],
        },
        base_dir=resolved_base_dir,
    )
    return {
        "target_month": next_month,
        "sample_size": len(invoices),
        "kaggle_dataset": dataset_cfg["kaggle_dataset"],
        "source_extract_dir": dataset_cfg["source_extract_dir"],
        "invoices": invoices,
    }
