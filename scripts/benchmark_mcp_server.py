from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "mcp_server" / "resources" / "sample_payloads" / "forecast_request.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.tools.forecast import forecast_sales
from mcp_server.tools.health import health_check
from mcp_server.tools.invoice import forecast_and_generate_invoice


def _load_request() -> dict[str, Any]:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _run_trials(
    name: str,
    fn: Callable[[], dict[str, Any]],
    repeats: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    successes = 0
    failures = 0
    timeout_failures = 0
    artifact_writes = 0
    metadata_writes = 0
    last_result: dict[str, Any] | None = None
    errors: list[str] = []

    for _ in range(repeats):
        started = perf_counter()
        try:
            result = fn()
            latencies_ms.append((perf_counter() - started) * 1000)
            successes += 1
            artifact_path = result.get("artifact_path")
            metadata_path = result.get("metadata_path")
            if artifact_path and Path(str(artifact_path)).exists():
                artifact_writes += 1
            if metadata_path and Path(str(metadata_path)).exists():
                metadata_writes += 1
            last_result = result
        except TimeoutError as error:
            latencies_ms.append((perf_counter() - started) * 1000)
            failures += 1
            timeout_failures += 1
            errors.append(str(error))
        except Exception as error:  # noqa: BLE001
            latencies_ms.append((perf_counter() - started) * 1000)
            failures += 1
            errors.append(f"{type(error).__name__}: {error}")

    return {
        "tool": name,
        "repeats": repeats,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": round(successes / repeats, 4) if repeats else 0.0,
        "timeout_rate": round(timeout_failures / repeats, 4) if repeats else 0.0,
        "artifact_write_rate": round(artifact_writes / repeats, 4) if repeats else 0.0,
        "metadata_write_rate": round(metadata_writes / repeats, 4) if repeats else 0.0,
        "latency_ms": {
            "mean": round(mean(latencies_ms), 2) if latencies_ms else 0.0,
            "p50": round(_percentile(latencies_ms, 0.5), 2) if latencies_ms else 0.0,
            "p95": round(_percentile(latencies_ms, 0.95), 2) if latencies_ms else 0.0,
        },
        "last_result": last_result,
        "errors": errors[:3],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark local MCP tool operations.")
    parser.add_argument("--health-repeats", type=int, default=3)
    parser.add_argument("--forecast-repeats", type=int, default=5)
    parser.add_argument("--invoice-repeats", type=int, default=1)
    parser.add_argument("--output", default="artifacts/metrics/mcp_benchmark.json")
    args = parser.parse_args(argv)

    request = _load_request()
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT / "artifacts" / "audit" / "audit.jsonl"
    audit_lines_before = 0
    if audit_path.exists():
        audit_lines_before = len(audit_path.read_text(encoding="utf-8").splitlines())

    results = {
        "benchmark_date": __import__("datetime").datetime.now().astimezone().isoformat(),
        "base_dir": str(ROOT),
        "sample_request": str(SAMPLE_PATH.relative_to(ROOT)),
        "runs": [
            _run_trials("health_check", lambda: health_check(base_dir=ROOT), args.health_repeats),
            _run_trials("forecast_sales", lambda: forecast_sales(request, base_dir=ROOT), args.forecast_repeats),
            _run_trials(
                "forecast_and_generate_invoice",
                lambda: forecast_and_generate_invoice(request, base_dir=ROOT),
                args.invoice_repeats,
            ),
        ],
    }
    audit_lines_after = 0
    if audit_path.exists():
        audit_lines_after = len(audit_path.read_text(encoding="utf-8").splitlines())
    results["audit_log_entries_added"] = audit_lines_after - audit_lines_before
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
