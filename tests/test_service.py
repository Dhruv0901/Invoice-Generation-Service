from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_server.services.feature_builder import build_features
from mcp_server.services.model_loader import load_model_bundle
from mcp_server.tools.forecast import forecast_sales
from mcp_server.tools.invoice import generate_invoice_docx, generate_invoices_from_kaggle_sample


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "mcp_server" / "resources" / "sample_payloads" / "forecast_request.json"


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MODEL_PATH"] = "tests/fixtures/missing-model.pkl"
        os.environ["FEATURE_MANIFEST_PATH"] = "models/feature_manifest.json"
        os.environ["METRICS_PATH"] = "reports/metrics.json"
        os.environ["ALLOW_HEURISTIC_MODEL"] = "true"
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

    def test_invoice_generation_writes_docx_and_metadata(self) -> None:
        result = generate_invoice_docx(self.request, base_dir=ROOT)
        artifact = Path(result["artifact_path"])
        metadata = Path(result["metadata_path"])
        self.assertTrue(artifact.exists())
        self.assertTrue(metadata.exists())

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
