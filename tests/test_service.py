from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_server.services.feature_builder import build_features
from mcp_server.services.model_loader import load_model_bundle
from mcp_server.services.qwen_invoice_renderer import _load_qwen_config, _qwen_invoice_lines
from mcp_server.tools.forecast import forecast_sales
from mcp_server.tools.invoice import generate_invoice_docx, generate_invoices_from_kaggle_sample


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "mcp_server" / "resources" / "sample_payloads" / "forecast_request.json"


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MODEL_PATH"] = "models/model.pkl"
        os.environ["FEATURE_MANIFEST_PATH"] = "models/feature_manifest.json"
        os.environ["METRICS_PATH"] = "reports/metrics.json"
        self.request = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        self.bundle = load_model_bundle(ROOT)

    def test_feature_order_matches_manifest(self) -> None:
        built = build_features(self.request, self.bundle.manifest)
        self.assertEqual(built.ordered_names, self.bundle.manifest["ordered_feature_names"])
        self.assertEqual(len(built.ordered_values), len(self.bundle.manifest["ordered_feature_names"]))

    def test_forecast_sales_returns_prediction(self) -> None:
        result = forecast_sales(self.request, base_dir=ROOT)
        self.assertGreaterEqual(result["predicted_monthly_sales"], 0.0)
        self.assertEqual(result["request_id"], "fcst_001")

    def test_load_model_bundle_requires_model_artifact(self) -> None:
        os.environ["MODEL_PATH"] = "tests/fixtures/missing-model.pkl"
        with self.assertRaises(FileNotFoundError):
            load_model_bundle(ROOT)

    @patch("mcp_server.services.qwen_invoice_renderer.urlopen")
    def test_invoice_generation_writes_docx_and_metadata(self, mock_urlopen) -> None:
        class SuccessResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"lines": ["Invoice", "Line 1"]})
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        mock_urlopen.return_value = SuccessResponse()
        result = generate_invoice_docx(self.request, base_dir=ROOT)
        artifact = Path(result["artifact_path"])
        metadata = Path(result["metadata_path"])
        self.assertTrue(artifact.exists())
        self.assertTrue(metadata.exists())

    def test_load_qwen_config_maps_hf_model_alias(self) -> None:
        temp_root = ROOT / "tests" / "_tmp_qwen_config"
        config_dir = temp_root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "qwen_config.json").write_text(
            json.dumps(
                {
                    "base_url": "https://router.huggingface.co/v1",
                    "api_key": "test-key",
                    "model": "qwen2.5:7b",
                }
            ),
            encoding="utf-8",
        )
        self.addCleanup(lambda: shutil.rmtree(temp_root, ignore_errors=True))
        config = _load_qwen_config(temp_root)
        self.assertEqual(config["base_url"], "https://router.huggingface.co/v1")
        self.assertEqual(config["model"], "Qwen/Qwen2.5-7B-Instruct")

    @patch("mcp_server.services.qwen_invoice_renderer.urlopen")
    def test_qwen_invoice_lines_requests_json_mode(self, mock_urlopen) -> None:
        class SuccessResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"lines": ["Invoice", "Line 1"]})
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        mock_urlopen.return_value = SuccessResponse()
        config = {
            "base_url": "https://example.com/v1",
            "api_key": "test-key",
            "model": "Qwen/Qwen2.5-7B-Instruct",
        }

        lines = _qwen_invoice_lines(
            {
                "invoice_number": "INV-1",
                "request_id": "REQ-1",
                "target_month": "2025-11-01",
                "product_id": "SKU-1",
                "product_name": "Alpha",
                "category": "Audio",
                "billed_quantity": 10,
                "unit_price": 9.99,
                "line_total": 99.9,
                "subtotal": 99.9,
                "tax": 10.0,
                "grand_total": 109.9,
            },
            config,
        )

        self.assertEqual(lines, ["Invoice", "Line 1"])
        request = mock_urlopen.call_args.args[0]
        self.assertIn('"response_format"', request.data.decode("utf-8"))

    @patch("mcp_server.tools.invoice.generate_invoice_docx")
    @patch("mcp_server.tools.invoice.build_invoice_requests")
    def test_generate_invoices_from_kaggle_sample_batches_requests(self, mock_build_requests, mock_generate_invoice) -> None:
        mock_build_requests.return_value = ([self.request, self.request], "2025-08")
        mock_generate_invoice.side_effect = [
            {"invoice_number": "INV-1", "artifact_path": "a.docx", "metadata_path": "a.json"},
            {"invoice_number": "INV-2", "artifact_path": "b.docx", "metadata_path": "b.json"},
        ]

        result = generate_invoices_from_kaggle_sample({"sample_size": 2}, base_dir=ROOT)

        self.assertEqual(result["sample_size"], 2)
        self.assertEqual(result["target_month"], "2025-08")
        self.assertEqual(len(result["invoices"]), 2)
        self.assertEqual(mock_generate_invoice.call_count, 2)


if __name__ == "__main__":
    unittest.main()
