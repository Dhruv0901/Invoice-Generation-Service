from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
FORECAST_CORE_ROOT = ROOT / "forecast_core"
FORECAST_CORE_SRC = FORECAST_CORE_ROOT / "src"
if str(FORECAST_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(FORECAST_CORE_SRC))

from ingest import (  # type: ignore  # noqa: E402
    build_monthly_sales_frame,
    clean_kaggle_sales,
    ensure_dataset,
    load_kaggle_source_tables,
    load_raw_data,
)


def load_params(params_path: Path) -> dict:
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_request_rows(
    params: dict,
    item_count: int,
    tax_rate: float,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    dataset_cfg = params.setdefault("dataset", {})
    dataset_cfg.setdefault("kaggle_dataset", "amirkhanh/synthetic-retail-dataset-1-2m-transactions")
    dataset_cfg.setdefault("source_extract_dir", str(FORECAST_CORE_ROOT / "data" / "raw" / "kaggle-synthetic-retail"))
    dataset_version = dataset_cfg.get("dataset_version", "kaggle-synthetic-retail")

    raw_paths = ensure_dataset(
        {
            "paths": {"raw_dir": str(FORECAST_CORE_ROOT / "data" / "raw")},
            "dataset_version": dataset_version,
            "dataset": {
                "source": dataset_cfg["source"],
                "kaggle_dataset": dataset_cfg["kaggle_dataset"],
                "source_extract_dir": dataset_cfg["source_extract_dir"],
                "product_limit": dataset_cfg.get("product_limit", 150),
            },
            "training": {"min_history_months": params["training"].get("min_history_months", 6)},
        }
    )
    raw_tables = load_raw_data(raw_paths)
    products = raw_tables["products"]
    inventory = raw_tables["inventory"]
    monthly_promotions = raw_tables["promotions"]

    source_dir = Path(dataset_cfg["source_extract_dir"])
    resolved_source_dir = source_dir if source_dir.is_absolute() else raw_paths.products.parent / source_dir
    source_tables = load_kaggle_source_tables(resolved_source_dir)
    sales = clean_kaggle_sales(source_tables["sales"])
    monthly_sales = build_monthly_sales_frame(sales)

    promo_sales = sales[sales["promo_id"].ne("")].copy() if "promo_id" in sales.columns else pd.DataFrame()
    if promo_sales.empty:
        promo_activity = pd.DataFrame(columns=["product_id", "month_start", "promo_day_count", "promo_event_count"])
    else:
        promo_sales["month_start"] = promo_sales["date"].dt.to_period("M").dt.to_timestamp()
        promo_activity = (
            promo_sales.groupby(["product_id", "month_start"], as_index=False)
            .agg(
                promo_day_count=("date", "nunique"),
                promo_event_count=("promo_id", "nunique"),
            )
        )

    latest_month = monthly_sales["month_start"].max()
    if pd.isna(latest_month):
        raise ValueError("No monthly sales periods were found in the Kaggle dataset.")
    next_month = (pd.Timestamp(latest_month) + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

    ordered = monthly_sales.sort_values(["product_id", "month_start"]).reset_index(drop=True)
    latest_rows = ordered[ordered["month_start"] == latest_month].copy()
    latest_rows["history_months"] = ordered.groupby("product_id")["month_start"].transform("count")
    eligible = latest_rows[latest_rows["history_months"] >= 6].copy()
    if eligible.empty:
        raise ValueError("No products have the required 6 months of history.")

    latest_ranked = eligible.drop_duplicates("product_id").copy()
    sample_size = min(item_count, len(latest_ranked))
    latest_ranked = latest_ranked.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    target_month = (pd.Timestamp(latest_month) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-01")

    for _, current_row in latest_ranked.iterrows():
        product_id = str(current_row["product_id"])
        history = (
            ordered[ordered["product_id"] == product_id]
            .sort_values("month_start", ascending=False)
            .head(6)
            .reset_index(drop=True)
        )
        if len(history) < 6:
            continue

        product = products[products["product_id"] == product_id].iloc[0]
        inventory_match = inventory[(inventory["product_id"] == product_id) & (inventory["month_start"] == latest_month)]
        promo_match = monthly_promotions[(monthly_promotions["product_id"] == product_id) & (monthly_promotions["month_start"] == latest_month)]
        promo_activity_match = promo_activity[(promo_activity["product_id"] == product_id) & (promo_activity["month_start"] == latest_month)]

        inventory_units = int(inventory_match["inventory_units"].iloc[0]) if not inventory_match.empty else 0
        promo_intensity = float(promo_match["promo_intensity"].iloc[0]) if not promo_match.empty else 0.0
        supplier_delay_flag = int(promo_match["supplier_delay_flag"].iloc[0]) if not promo_match.empty else 0
        promo_day_count = float(promo_activity_match["promo_day_count"].iloc[0]) if not promo_activity_match.empty else 0.0
        promo_event_count = float(promo_activity_match["promo_event_count"].iloc[0]) if not promo_activity_match.empty else 0.0

        row = {
            "request_id": f"kaggle_fcst_{len(rows) + 1:03d}",
            "target_month": target_month,
            "product_id": product_id,
            "product_name": str(product["product_name"]),
            "category": str(product["category"]),
            "base_price": round(float(product["base_price"]), 2),
            "trend_strength": round(float(product["trend_strength"]), 4),
            "seasonality_strength": round(float(product["seasonality_strength"]), 3),
            "monthly_sales_lag_1": int(history.iloc[0]["monthly_sales"]),
            "monthly_sales_lag_2": int(history.iloc[1]["monthly_sales"]),
            "monthly_sales_lag_3": int(history.iloc[2]["monthly_sales"]),
            "monthly_sales_lag_4": int(history.iloc[3]["monthly_sales"]),
            "monthly_sales_lag_5": int(history.iloc[4]["monthly_sales"]),
            "monthly_sales_lag_6": int(history.iloc[5]["monthly_sales"]),
            "average_price": round(float(current_row["average_price"]), 2),
            "inventory_units": inventory_units,
            "promo_intensity": round(promo_intensity, 4),
            "supplier_delay_flag": supplier_delay_flag,
            "promo_day_count": promo_day_count,
            "promo_event_count": promo_event_count,
            "active_days": int(current_row["active_days"]),
            "tax_rate": tax_rate,
        }
        rows.append(row)

    if not rows:
        raise ValueError("No forecast rows were generated after filtering for 6-month history.")

    return pd.DataFrame(rows), next_month


def build_invoice_requests(
    params: dict,
    item_count: int = 30,
    tax_rate: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[dict[str, Any]], str]:
    frame, next_month = build_request_rows(
        params=params,
        item_count=item_count,
        tax_rate=tax_rate,
        random_seed=random_seed,
    )
    requests: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        requests.append(
            {
                "request_id": row["request_id"],
                "target_month": row["target_month"],
                "product_id": row["product_id"],
                "product_context": {
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "base_price": row["base_price"],
                    "trend_strength": row["trend_strength"],
                    "seasonality_strength": row["seasonality_strength"],
                },
                "history": {
                    f"monthly_sales_lag_{index}": row[f"monthly_sales_lag_{index}"]
                    for index in range(1, 7)
                },
                "current_context": {
                    "average_price": row["average_price"],
                    "inventory_units": row["inventory_units"],
                    "promo_intensity": row["promo_intensity"],
                    "supplier_delay_flag": row["supplier_delay_flag"],
                    "promo_day_count": row["promo_day_count"],
                    "promo_event_count": row["promo_event_count"],
                    "active_days": row["active_days"],
                },
                "tax_rate": row["tax_rate"],
                "provenance": {
                    "dataset": params["dataset"]["kaggle_dataset"],
                    "sampling": "random_eligible_products",
                    "sample_size": item_count,
                    "random_seed": random_seed,
                },
            }
        )
    return requests, next_month


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate next-month forecast request rows from Kaggle source data.")
    parser.add_argument("--count", type=int, default=30, help="Number of forecast items to generate.")
    parser.add_argument("--tax-rate", type=float, default=0.1, help="Tax rate to include in the request rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for row sampling.")
    parser.add_argument(
        "--source-dir",
        default=str(FORECAST_CORE_ROOT / "data" / "raw" / "kaggle-synthetic-retail"),
        help="Directory where the Kaggle dataset will be extracted or reused.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT),
        help="Directory where {next_month}_forecast.csv will be written.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "json"),
        default="csv",
        help="Write flattened forecast rows as CSV or MCP request payloads as JSON.",
    )
    args = parser.parse_args(argv)

    params = load_params(FORECAST_CORE_ROOT / "params.yaml")
    params["dataset"]["source_extract_dir"] = args.source_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format == "json":
        requests, next_month = build_invoice_requests(
            params=params,
            item_count=args.count,
            tax_rate=args.tax_rate,
            random_seed=args.seed,
        )
        output_path = output_dir / f"{next_month}_forecast_requests.json"
        output_path.write_text(json.dumps(requests, indent=2), encoding="utf-8")
    else:
        frame, next_month = build_request_rows(
            params=params,
            item_count=args.count,
            tax_rate=args.tax_rate,
            random_seed=args.seed,
        )
        output_path = output_dir / f"{next_month}_forecast.csv"
        frame.to_csv(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
