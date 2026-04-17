from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.sax.saxutils import escape


def _document_xml(lines: list[str]) -> str:
    paragraphs = []
    for line in lines:
        safe = escape(line)
        paragraphs.append(
            "<w:p><w:r><w:t xml:space=\"preserve\">"
            f"{safe}"
            "</w:t></w:r></w:p>"
        )
    body = "".join(paragraphs)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        f"<w:body>{body}<w:sectPr/></w:body>"
        "</w:document>"
    )


def _write_minimal_docx(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
                "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
                "<Override PartName=\"/word/document.xml\" "
                "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                "<Relationship Id=\"rId1\" "
                "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
                "Target=\"word/document.xml\"/>"
                "</Relationships>"
            ),
        )
        archive.writestr("word/document.xml", _document_xml(lines))


def _load_qwen_config(base_dir: Path | None = None) -> dict[str, str]:
    root = base_dir or Path.cwd()
    config_path = root / "config" / "qwen_config.json"
    config: dict[str, str] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    base_url = str(config.get("base_url", os.getenv("QWEN_BASE_URL", ""))).strip()
    api_key = str(config.get("api_key", os.getenv("QWEN_API_KEY", ""))).strip()
    model = str(config.get("model", os.getenv("QWEN_MODEL", "qwen2.5"))).strip()
    return {"base_url": base_url.rstrip("/"), "api_key": api_key, "model": model}


def _default_invoice_lines(invoice_payload: dict[str, Any]) -> list[str]:
    return [
        "Forecast-Based Pro Forma Invoice",
        "",
        f"Invoice Number: {invoice_payload['invoice_number']}",
        f"Request ID: {invoice_payload['request_id']}",
        f"Target Month: {invoice_payload['target_month']}",
        "",
        f"Product ID: {invoice_payload['product_id']}",
        f"Product Name: {invoice_payload['product_name']}",
        f"Category: {invoice_payload['category']}",
        "",
        f"Forecast Quantity: {invoice_payload['billed_quantity']}",
        f"Unit Price: {invoice_payload['unit_price']:.2f}",
        f"Line Total: {invoice_payload['line_total']:.2f}",
        f"Subtotal: {invoice_payload['subtotal']:.2f}",
        f"Tax: {invoice_payload['tax']:.2f}",
        f"Grand Total: {invoice_payload['grand_total']:.2f}",
        "",
        "Disclaimer: This document is generated from forecasted demand and does not represent a confirmed sale.",
    ]


def _qwen_invoice_lines(invoice_payload: dict[str, Any], config: dict[str, str]) -> list[str]:
    prompt = (
        "Create concise plain-text invoice lines for a forecast-based pro forma invoice. "
        "Return JSON with a single key `lines` containing an array of short strings. "
        "Do not add markdown or code fences.\n\n"
        f"Invoice payload:\n{json.dumps(invoice_payload, ensure_ascii=True)}"
    )
    body = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate structured business invoice text. "
                    "Keep it factual, compact, and suitable for a Word document."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }

    request = Request(
        f"{config['base_url']}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key'] or 'local'}",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    lines = parsed.get("lines", [])
    if not isinstance(lines, list) or not lines:
        raise ValueError("Qwen returned an invalid invoice payload.")
    return [str(line) for line in lines]


def render_invoice_docx(output_path: Path, invoice_payload: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    config = _load_qwen_config(base_dir)
    qwen_configured = bool(config["base_url"])
    qwen_used = False
    lines = _default_invoice_lines(invoice_payload)
    render_error = ""

    if qwen_configured:
        try:
            lines = _qwen_invoice_lines(invoice_payload, config)
            qwen_used = True
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
            render_error = str(error)

    _write_minimal_docx(output_path, lines)
    return {
        "used_qwen": qwen_used,
        "qwen_configured": qwen_configured,
        "qwen_model": config["model"],
        "qwen_base_url": config["base_url"],
        "render_fallback": "deterministic_docx" if not qwen_used else "",
        "render_error": render_error,
    }
