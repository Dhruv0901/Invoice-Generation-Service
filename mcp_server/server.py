from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from mcp_server.tools.forecast import forecast_sales
from mcp_server.tools.health import health_check
from mcp_server.tools.invoice import (
    forecast_and_generate_invoice,
    generate_invoice_docx,
    generate_invoices_from_kaggle_sample,
)
from mcp_server.tools.metadata import get_model_metadata


ToolHandler = Callable[..., dict[str, Any]]


TOOLS: dict[str, ToolHandler] = {
    "health_check": health_check,
    "get_model_metadata": get_model_metadata,
    "forecast_sales": forecast_sales,
    "generate_invoice_docx": generate_invoice_docx,
    "forecast_and_generate_invoice": forecast_and_generate_invoice,
    "generate_invoices_from_kaggle_sample": generate_invoices_from_kaggle_sample,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _invoke(tool_name: str, payload: dict[str, Any] | None, base_dir: Path) -> dict[str, Any]:
    handler = TOOLS[tool_name]
    if payload is None:
        return handler(base_dir=base_dir)
    return handler(payload, base_dir=base_dir)


def run_stdio(base_dir: Path) -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request = json.loads(line)
        tool_name = request["tool"]
        payload = request.get("input")
        response = _invoke(tool_name, payload, base_dir)
        sys.stdout.write(json.dumps({"tool": tool_name, "result": response}) + "\n")
        sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Future sales forecast MCP-style server")
    parser.add_argument("tool", nargs="?", choices=sorted(TOOLS.keys()))
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--stdio", action="store_true")
    args = parser.parse_args(argv)

    base_dir = Path.cwd()

    if args.stdio:
        return run_stdio(base_dir)

    if not args.tool:
        parser.error("either --stdio or a tool name is required")

    payload = _load_json(Path(args.input_path)) if args.input_path else None
    result = _invoke(args.tool, payload, base_dir)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
