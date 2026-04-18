from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.generate_next_month_forecast import build_invoice_requests, build_request_rows


ROOT = Path(__file__).resolve().parents[1]


class ForecastPipelineTests(unittest.TestCase):
    def test_build_request_rows_from_minimal_kaggle_shapes(self) -> None:
        source_dir = ROOT / "tests" / "_tmp_kaggle_source"
        source_dir.mkdir(parents=True, exist_ok=True)

        products = pd.DataFrame(
            [
                {"sku_id": "SKU-1", "sku_name": "Alpha Headphones", "category": "Audio", "unit_price": 99.0},
                {"sku_id": "SKU-2", "sku_name": "Beta Mouse", "category": "Accessories", "unit_price": 39.0},
            ]
        )
        stores = pd.DataFrame([{"store_id": "S1", "store_name": "Main", "city": "Sydney", "store_type": "Flagship", "opening_date": "2024-01-01"}])
        promotions = pd.DataFrame(
            [
                {"promo_id": "P1", "start_date": "2025-06-01", "end_date": "2025-06-07", "discount_pct": 15, "target_type": "sku", "target_value": "SKU-1"},
            ]
        )
        inventory = pd.DataFrame(
            [
                {"store_id": "S1", "sku_id": "SKU-1", "stock_on_hand": 500, "reorder_point": 30, "safety_stock": 40, "last_restock_date": "2025-06-01"},
                {"store_id": "S1", "sku_id": "SKU-2", "stock_on_hand": 400, "reorder_point": 20, "safety_stock": 30, "last_restock_date": "2025-06-01"},
            ]
        )

        sales_rows = []
        months = pd.date_range("2025-01-01", periods=7, freq="MS")
        for month_index, month_start in enumerate(months, start=1):
            for day in (1, 8, 15):
                sales_rows.append(
                    {
                        "date": (month_start + pd.Timedelta(days=day - 1)).strftime("%Y-%m-%d"),
                        "receipt_id": f"R-A-{month_index}-{day}",
                        "store_id": "S1",
                        "sku_id": "SKU-1",
                        "customer_id": f"C-{month_index}-{day}",
                        "quantity": 20 + month_index,
                        "unit_price": 95.0,
                        "total_value": (20 + month_index) * 95.0,
                        "channel": "store",
                        "promo_id": "P1" if month_index == 6 and day == 1 else "",
                    }
                )
                sales_rows.append(
                    {
                        "date": (month_start + pd.Timedelta(days=day - 1)).strftime("%Y-%m-%d"),
                        "receipt_id": f"R-B-{month_index}-{day}",
                        "store_id": "S1",
                        "sku_id": "SKU-2",
                        "customer_id": f"D-{month_index}-{day}",
                        "quantity": 12 + month_index,
                        "unit_price": 38.0,
                        "total_value": (12 + month_index) * 38.0,
                        "channel": "store",
                        "promo_id": "",
                    }
                )
        sales = pd.DataFrame(sales_rows)

        stores.to_csv(source_dir / "store_master.csv", index=False)
        products.to_csv(source_dir / "sku_master.csv", index=False)
        promotions.to_csv(source_dir / "promotions.csv", index=False)
        inventory.to_csv(source_dir / "inventory_snapshot.csv", index=False)
        sales.to_csv(source_dir / "sales_transactions.csv", index=False)

        params = {
            "dataset": {
                "product_limit": 150,
                "source": "synthetic",
                "kaggle_dataset": "synthetic",
                "source_extract_dir": str(source_dir),
                "dataset_version": "test",
            },
            "training": {"min_history_months": 6},
        }

        frame, next_month = build_request_rows(params, item_count=2, tax_rate=0.1)
        self.assertEqual(len(frame), 2)
        self.assertEqual(next_month, "2025-08")
        self.assertEqual(frame.loc[0, "target_month"], "2025-08-01")
        self.assertIn("monthly_sales_lag_6", frame.columns)
        self.assertIn("promo_day_count", frame.columns)

    @patch("scripts.generate_next_month_forecast.build_request_rows")
    def test_build_invoice_requests_shapes_mcp_payloads(self, mock_build_request_rows) -> None:
        mock_build_request_rows.return_value = (
            pd.DataFrame(
                [
                    {
                        "request_id": "kaggle_fcst_001",
                        "target_month": "2025-08-01",
                        "product_id": "SKU-1",
                        "product_name": "Alpha Headphones",
                        "category": "Audio",
                        "base_price": 99.0,
                        "trend_strength": 0.2,
                        "seasonality_strength": 0.1,
                        "monthly_sales_lag_1": 10,
                        "monthly_sales_lag_2": 9,
                        "monthly_sales_lag_3": 8,
                        "monthly_sales_lag_4": 7,
                        "monthly_sales_lag_5": 6,
                        "monthly_sales_lag_6": 5,
                        "average_price": 95.0,
                        "inventory_units": 100,
                        "promo_intensity": 0.15,
                        "supplier_delay_flag": 0,
                        "promo_day_count": 2,
                        "promo_event_count": 1,
                        "active_days": 30,
                        "tax_rate": 0.1,
                    }
                ]
            ),
            "2025-08",
        )
        params = {
            "dataset": {"kaggle_dataset": "synthetic"},
            "training": {"min_history_months": 6},
        }

        requests, next_month = build_invoice_requests(params, item_count=30, tax_rate=0.1, random_seed=7)

        self.assertEqual(next_month, "2025-08")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["product_context"]["product_name"], "Alpha Headphones")
        self.assertEqual(requests[0]["history"]["monthly_sales_lag_6"], 5)
        self.assertEqual(requests[0]["current_context"]["promo_event_count"], 1)
        self.assertEqual(requests[0]["provenance"]["sample_size"], 30)

    def test_build_request_rows_from_bm_kaggle_shapes(self) -> None:
        source_dir = ROOT / "tests" / "_tmp_kaggle_source_bm"
        source_dir.mkdir(parents=True, exist_ok=True)

        products = pd.DataFrame(
            [
                {"sku_id": "SKU-1", "sku_name": "Alpha Headphones", "category": "Audio", "base_price": 99.0},
                {"sku_id": "SKU-2", "sku_name": "Beta Mouse", "category": "Accessories", "base_price": 39.0},
            ]
        )
        stores = pd.DataFrame([{"store_id": "S1", "store_name": "Main"}])
        promotions = pd.DataFrame(
            [
                {"promo_id": 1, "promo_name": "Holiday Sale", "start_date": "2025-06-01", "end_date": "2025-06-07", "discount_pct": 15, "promo_type": "Seasonal"},
            ]
        )
        inventory = pd.DataFrame(
            [
                {"store_id": "S1", "sku_id": "SKU-1", "stock_on_hand": 500, "reorder_point": 30, "safety_stock": 40},
                {"store_id": "S1", "sku_id": "SKU-2", "stock_on_hand": 400, "reorder_point": 20, "safety_stock": 30},
            ]
        )

        sales_rows = []
        months = pd.date_range("2025-01-01", periods=7, freq="MS")
        for month_index, month_start in enumerate(months, start=1):
            for day in (1, 8, 15):
                sales_rows.append(
                    {
                        "date": (month_start + pd.Timedelta(days=day - 1)).strftime("%Y-%m-%d"),
                        "store_id": "S1",
                        "sku_id": "SKU-1",
                        "customer_id": f"C-{month_index}-{day}",
                        "quantity": 20 + month_index,
                        "unit_price": 95.0,
                        "total_value": (20 + month_index) * 95.0,
                        "channel": "store",
                        "discount_pct": 15.0 if month_index == 6 and day == 1 else 0.0,
                    }
                )
                sales_rows.append(
                    {
                        "date": (month_start + pd.Timedelta(days=day - 1)).strftime("%Y-%m-%d"),
                        "store_id": "S1",
                        "sku_id": "SKU-2",
                        "customer_id": f"D-{month_index}-{day}",
                        "quantity": 12 + month_index,
                        "unit_price": 38.0,
                        "total_value": (12 + month_index) * 38.0,
                        "channel": "store",
                        "discount_pct": 0.0,
                    }
                )
        sales = pd.DataFrame(sales_rows)

        stores.to_csv(source_dir / "bm_stores.csv", index=False)
        products.to_csv(source_dir / "bm_skus.csv", index=False)
        promotions.to_csv(source_dir / "bm_promotions.csv", index=False)
        inventory.to_csv(source_dir / "bm_inventory.csv", index=False)
        sales.to_csv(source_dir / "bm_sales.csv", index=False)

        params = {
            "dataset": {
                "product_limit": 150,
                "source": "synthetic",
                "kaggle_dataset": "synthetic",
                "source_extract_dir": str(source_dir),
                "dataset_version": "test-bm",
            },
            "training": {"min_history_months": 6},
        }

        frame, next_month = build_request_rows(params, item_count=2, tax_rate=0.1)
        self.assertEqual(len(frame), 2)
        self.assertEqual(next_month, "2025-08")
        self.assertIn("promo_intensity", frame.columns)
        self.assertGreaterEqual(float(frame["promo_intensity"].max()), 0.0)

    @patch("scripts.generate_next_month_forecast.load_raw_data")
    @patch("scripts.generate_next_month_forecast.load_kaggle_source_tables")
    @patch("scripts.generate_next_month_forecast.ensure_dataset")
    def test_build_request_rows_filters_to_available_products(
        self,
        mock_ensure_dataset,
        mock_load_kaggle_source_tables,
        mock_load_raw_data,
    ) -> None:
        mock_ensure_dataset.return_value = type("RawPathsStub", (), {"products": ROOT / "forecast_core" / "data" / "raw" / "products.csv"})()
        mock_load_raw_data.return_value = {
            "products": pd.DataFrame(
                [
                    {
                        "product_id": "SKU-1",
                        "product_name": "Alpha Headphones",
                        "category": "Audio",
                        "base_price": 99.0,
                        "trend_strength": 0.1,
                        "seasonality_strength": 0.2,
                    }
                ]
            ),
            "inventory": pd.DataFrame([{"product_id": "SKU-1", "month_start": pd.Timestamp("2025-07-01"), "inventory_units": 50}]),
            "promotions": pd.DataFrame([{"product_id": "SKU-1", "month_start": pd.Timestamp("2025-07-01"), "promo_intensity": 0.1, "supplier_delay_flag": 0}]),
        }

        sales_rows = []
        months = pd.date_range("2025-01-01", periods=7, freq="MS")
        for month_start in months:
            sales_rows.append({"date": month_start.strftime("%Y-%m-%d"), "sku_id": "SKU-1", "quantity": 10, "unit_price": 20.0, "discount_pct": 0.0})
            sales_rows.append({"date": month_start.strftime("%Y-%m-%d"), "sku_id": "SKU-2", "quantity": 10, "unit_price": 20.0, "discount_pct": 0.0})
        mock_load_kaggle_source_tables.return_value = {
            "sales": pd.DataFrame(sales_rows),
        }

        params = {
            "dataset": {
                "product_limit": 1,
                "source": "synthetic",
                "kaggle_dataset": "synthetic",
                "source_extract_dir": str(ROOT / "tests" / "_tmp_kaggle_source"),
                "dataset_version": "test-filtered",
            },
            "training": {"min_history_months": 6},
        }

        frame, _ = build_request_rows(params, item_count=2, tax_rate=0.1)
        self.assertEqual(frame["product_id"].tolist(), ["SKU-1"])


if __name__ == "__main__":
    unittest.main()
