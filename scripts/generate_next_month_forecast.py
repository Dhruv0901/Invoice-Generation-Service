from __future__ import annotations

import argparse
import base64
import json
import random
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAGGLE_DATASET = "amirkhanh/synthetic-retail-dataset-1-2m-transactions"
DEFAULT_SOURCE_DIR = ROOT / "data" / "raw" / "kaggle-synthetic-retail"
DEFAULT_RAW_DIR = ROOT / "data" / "prepared"

KAGGLE_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "stores": ("store_master.csv", "bm_stores.csv"),
    "products": ("sku_master.csv", "bm_skus.csv"),
    "sales": ("sales_transactions.csv", "bm_sales.csv"),
    "inventory": ("inventory_snapshot.csv", "bm_inventory.csv"),
    "promotions": ("promotions.csv", "bm_promotions.csv"),
}


@dataclass(frozen=True)
class RawPaths:
    products: Path
    sales: Path
    inventory: Path
    promotions: Path


def default_params(source_dir: str | Path | None = None) -> dict[str, Any]:
    resolved_source_dir = Path(source_dir) if source_dir is not None else DEFAULT_SOURCE_DIR
    return {
        "dataset": {
            "source": "synthetic",
            "kaggle_dataset": DEFAULT_KAGGLE_DATASET,
            "source_extract_dir": str(resolved_source_dir),
            "dataset_version": "kaggle-synthetic-retail",
            "product_limit": 150,
        },
        "training": {"min_history_months": 6},
    }


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def dataset_is_current(raw_paths: RawPaths, metadata_path: Path, dataset_version: str) -> bool:
    if not all(path.exists() for path in raw_paths.__dict__.values()):
        return False
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metadata.get("dataset_version") == dataset_version


def ensure_dataset(params: dict[str, Any]) -> RawPaths:
    raw_dir = Path(params["paths"]["raw_dir"])
    ensure_directories([raw_dir])
    raw_paths = RawPaths(
        products=raw_dir / "products.csv",
        sales=raw_dir / "sales.csv",
        inventory=raw_dir / "inventory.csv",
        promotions=raw_dir / "promotions.csv",
    )
    metadata_path = raw_dir / "_dataset_metadata.json"

    if dataset_is_current(raw_paths, metadata_path, params["dataset_version"]):
        return raw_paths

    prepare_kaggle_dataset(raw_paths, metadata_path, params)
    return raw_paths


def prepare_kaggle_dataset(paths: RawPaths, metadata_path: Path, params: dict[str, Any]) -> None:
    dataset_cfg = params["dataset"]
    source_dir = Path(dataset_cfg["source_extract_dir"])
    resolved_source_dir = source_dir if source_dir.is_absolute() else paths.products.parent / source_dir
    resolved_source_dir.mkdir(parents=True, exist_ok=True)

    ensure_kaggle_source_files(resolved_source_dir, dataset_cfg["kaggle_dataset"])

    source_tables = load_kaggle_source_tables(resolved_source_dir)
    raw_tables = transform_kaggle_retail_to_raw_tables(source_tables, params)

    raw_tables["products"].to_csv(paths.products, index=False)
    raw_tables["sales"].to_csv(paths.sales, index=False)
    raw_tables["inventory"].to_csv(paths.inventory, index=False)
    raw_tables["promotions"].to_csv(paths.promotions, index=False)

    metadata = {
        "dataset_version": params["dataset_version"],
        "source": dataset_cfg["source"],
        "kaggle_dataset": dataset_cfg["kaggle_dataset"],
        "source_extract_dir": str(resolved_source_dir),
        "product_limit": int(dataset_cfg.get("product_limit", 150)),
        "min_history_months": int(params["training"].get("min_history_months", 6)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def ensure_kaggle_source_files(source_dir: Path, kaggle_dataset: str) -> None:
    if _source_layout_is_supported(source_dir):
        return

    dataset_slug = kaggle_dataset.split("/")[-1]
    archive_path = source_dir / f"{dataset_slug}.zip"
    if not archive_path.exists():
        download_kaggle_archive(kaggle_dataset, archive_path, ROOT)
    extract_kaggle_archive(archive_path, source_dir)

    if not _source_layout_is_supported(source_dir):
        missing = sorted(
            canonical_name
            for canonical_name, aliases in _canonical_source_file_map().items()
            if not any((source_dir / alias).exists() for alias in aliases)
        )
        raise FileNotFoundError(
            "Kaggle dataset archive was downloaded, but the expected CSV files were still missing. "
            f"Missing: {', '.join(missing)}"
        )


def download_kaggle_archive(kaggle_dataset: str, archive_path: Path, project_root: Path) -> None:
    credentials = load_kaggle_credentials(project_root)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    encoded_ref = quote(kaggle_dataset, safe="/")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{encoded_ref}"
    token = base64.b64encode(f"{credentials['username']}:{credentials['key']}".encode("utf-8")).decode("ascii")
    request = Request(
        url,
        headers={
            "Authorization": f"Basic {token}",
            "User-Agent": "invoice-generation-service/1.0",
        },
    )

    try:
        with urlopen(request) as response, archive_path.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle)
    except HTTPError as error:
        if archive_path.exists():
            archive_path.unlink()
        raise RuntimeError(
            f"Kaggle download failed with HTTP {error.code}. Check the dataset slug and your Kaggle credentials."
        ) from error
    except URLError as error:
        if archive_path.exists():
            archive_path.unlink()
        raise RuntimeError("Kaggle download failed due to a network error.") from error

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        raise FileNotFoundError(f"Kaggle download completed but archive was not found at {archive_path}")


def load_kaggle_credentials(project_root: Path) -> dict[str, str]:
    username = ""
    key = ""

    config_candidates = [
        project_root / "config" / "kaggle_credentials.json",
        project_root / ".secrets" / "kaggle_credentials.json",
    ]
    for candidate in config_candidates:
        if candidate.exists():
            config = json.loads(candidate.read_text(encoding="utf-8"))
            username = str(config.get("username", "")).strip()
            key = str(config.get("key", "")).strip()
            if username and key:
                break

    if not username:
        username = str(os.environ.get("KAGGLE_USERNAME", "")).strip()
    if not key:
        key = str(os.environ.get("KAGGLE_KEY", "")).strip()

    if username and key:
        return {"username": username, "key": key}

    raise RuntimeError(
        "Kaggle credentials were not found. Add them to `config/kaggle_credentials.json` "
        "or set `KAGGLE_USERNAME` and `KAGGLE_KEY`."
    )


def extract_kaggle_archive(archive_path: Path, source_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(source_dir)


def _canonical_source_file_map() -> dict[str, tuple[str, ...]]:
    return {aliases[0]: aliases for aliases in KAGGLE_SOURCE_ALIASES.values()}


def _source_layout_is_supported(source_dir: Path) -> bool:
    return all(_resolve_source_file(source_dir, logical_name) is not None for logical_name in KAGGLE_SOURCE_ALIASES)


def _resolve_source_file(source_dir: Path, logical_name: str) -> Path | None:
    for candidate_name in KAGGLE_SOURCE_ALIASES[logical_name]:
        candidate_path = source_dir / candidate_name
        if candidate_path.exists():
            return candidate_path
    return None


def load_kaggle_source_tables(source_dir: Path) -> dict[str, pd.DataFrame]:
    resolved_files = {logical_name: _resolve_source_file(source_dir, logical_name) for logical_name in KAGGLE_SOURCE_ALIASES}
    missing = [
        aliases[0]
        for logical_name, aliases in KAGGLE_SOURCE_ALIASES.items()
        if resolved_files[logical_name] is None
    ]
    if missing:
        raise FileNotFoundError(
            "Kaggle source files are missing. Expected them under "
            f"{source_dir}. Missing: {', '.join(missing)}."
        )

    return {
        "stores": pd.read_csv(resolved_files["stores"]),
        "products": pd.read_csv(resolved_files["products"]),
        "sales": pd.read_csv(resolved_files["sales"]),
        "inventory": pd.read_csv(resolved_files["inventory"]),
        "promotions": pd.read_csv(resolved_files["promotions"]),
    }


def clean_kaggle_sales(source_frame: pd.DataFrame) -> pd.DataFrame:
    sales = source_frame.rename(
        columns={
            "transaction_id": "transaction_id",
            "store_id": "store_id",
            "sku_id": "product_id",
            "product_id": "product_id",
            "date": "date",
            "transaction_date": "date",
            "quantity": "quantity",
            "qty": "quantity",
            "unit_price": "unit_price",
            "price": "unit_price",
            "discount": "discount_amount",
            "discount_amount": "discount_amount",
            "discount_pct": "discount_pct",
            "discount_percent": "discount_pct",
            "promo_id": "promo_id",
            "promotion_id": "promo_id",
        }
    ).copy()
    required = {"product_id", "date", "quantity", "unit_price"}
    missing = required.difference(sales.columns)
    if missing:
        raise ValueError(f"Kaggle sales source is missing columns: {sorted(missing)}")

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce")
    sales["unit_price"] = pd.to_numeric(sales["unit_price"], errors="coerce")
    discount_series = sales["discount_amount"] if "discount_amount" in sales.columns else pd.Series(0.0, index=sales.index)
    discount_pct_series = sales["discount_pct"] if "discount_pct" in sales.columns else pd.Series(0.0, index=sales.index)
    promo_series = sales["promo_id"] if "promo_id" in sales.columns else pd.Series("", index=sales.index, dtype="string")
    sales["discount_amount"] = pd.to_numeric(discount_series, errors="coerce").fillna(0.0)
    sales["discount_pct"] = pd.to_numeric(discount_pct_series, errors="coerce").fillna(0.0)
    sales["promo_id"] = promo_series.astype("string").fillna("")
    sales["product_id"] = sales["product_id"].astype(str).str.strip()

    sales = sales[sales["date"].notna() & sales["quantity"].gt(0) & sales["unit_price"].gt(0)].copy()
    sales["date"] = sales["date"].dt.normalize()
    sales["sales_value"] = (sales["unit_price"] * sales["quantity"]) - sales["discount_amount"]
    sales["sales_value"] = sales["sales_value"].clip(lower=0.0)
    return sales.reset_index(drop=True)


def clean_kaggle_products(source_frame: pd.DataFrame) -> pd.DataFrame:
    products = source_frame.rename(
        columns={
            "sku_id": "product_id",
            "product_id": "product_id",
            "sku_name": "product_name",
            "product_name": "product_name",
            "category": "category",
            "department": "category",
            "base_price": "base_price",
            "list_price": "base_price",
        }
    ).copy()
    required = {"product_id", "product_name"}
    missing = required.difference(products.columns)
    if missing:
        raise ValueError(f"Kaggle product source is missing columns: {sorted(missing)}")

    products["product_id"] = products["product_id"].astype(str).str.strip()
    products["product_name"] = products["product_name"].astype("string").fillna("").str.strip()
    category_series = products["category"] if "category" in products.columns else pd.Series("General Merchandise", index=products.index)
    base_price_series = products["base_price"] if "base_price" in products.columns else pd.Series(float("nan"), index=products.index)
    products["category"] = category_series.astype("string").fillna("General Merchandise").str.strip()
    products["base_price"] = pd.to_numeric(base_price_series, errors="coerce")
    return products[products["product_name"].ne("")].drop_duplicates("product_id").reset_index(drop=True)


def clean_kaggle_inventory(source_frame: pd.DataFrame) -> pd.DataFrame:
    inventory = source_frame.rename(
        columns={
            "sku_id": "product_id",
            "product_id": "product_id",
            "stock_on_hand": "stock_on_hand",
            "inventory_units": "stock_on_hand",
            "reorder_point": "reorder_point",
            "safety_stock": "safety_stock",
        }
    ).copy()
    required = {"product_id", "stock_on_hand"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"Kaggle inventory source is missing columns: {sorted(missing)}")

    inventory["product_id"] = inventory["product_id"].astype(str).str.strip()
    inventory["stock_on_hand"] = pd.to_numeric(inventory["stock_on_hand"], errors="coerce").fillna(0.0)
    reorder_series = inventory["reorder_point"] if "reorder_point" in inventory.columns else pd.Series(0.0, index=inventory.index)
    safety_series = inventory["safety_stock"] if "safety_stock" in inventory.columns else pd.Series(0.0, index=inventory.index)
    inventory["reorder_point"] = pd.to_numeric(reorder_series, errors="coerce").fillna(0.0)
    inventory["safety_stock"] = pd.to_numeric(safety_series, errors="coerce").fillna(0.0)
    return inventory.reset_index(drop=True)


def clean_kaggle_promotions(source_frame: pd.DataFrame) -> pd.DataFrame:
    promotions = source_frame.rename(
        columns={
            "promo_id": "promo_id",
            "promotion_id": "promo_id",
            "start_date": "start_date",
            "end_date": "end_date",
            "discount_pct": "discount_pct",
            "discount_percent": "discount_pct",
            "target_type": "target_type",
            "target_value": "target_value",
        }
    ).copy()
    required = {"promo_id", "start_date", "end_date"}
    missing = required.difference(promotions.columns)
    if missing:
        raise ValueError(f"Kaggle promotions source is missing columns: {sorted(missing)}")

    promotions["promo_id"] = promotions["promo_id"].astype("string").fillna("").str.strip()
    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce")
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce")
    discount_series = promotions["discount_pct"] if "discount_pct" in promotions.columns else pd.Series(0.0, index=promotions.index)
    target_type_series = promotions["target_type"] if "target_type" in promotions.columns else pd.Series("", index=promotions.index, dtype="string")
    target_value_series = promotions["target_value"] if "target_value" in promotions.columns else pd.Series("", index=promotions.index, dtype="string")
    promotions["discount_pct"] = pd.to_numeric(discount_series, errors="coerce").fillna(0.0)
    promotions["target_type"] = target_type_series.astype("string").fillna("").str.strip().str.lower()
    promotions["target_value"] = target_value_series.astype("string").fillna("").str.strip()
    return promotions[promotions["promo_id"].ne("") & promotions["start_date"].notna() & promotions["end_date"].notna()].reset_index(drop=True)


def build_monthly_sales_frame(sales: pd.DataFrame) -> pd.DataFrame:
    return (
        sales.assign(month_start=sales["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["product_id", "month_start"], as_index=False)
        .agg(
            monthly_sales=("quantity", "sum"),
            average_price=("unit_price", "mean"),
            sales_value=("sales_value", "sum"),
            active_days=("date", lambda values: values.nunique()),
        )
        .sort_values(["product_id", "month_start"])
        .reset_index(drop=True)
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _trend_strength(series: pd.Series) -> float:
    values = [float(value) for value in series.tolist()]
    count = len(values)
    if count <= 1:
        return 0.0
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    if denominator == 0 or mean_y == 0:
        return 0.0
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    return numerator / denominator / mean_y


def build_products_table(products: pd.DataFrame, monthly_sales: pd.DataFrame) -> pd.DataFrame:
    stats: list[dict[str, Any]] = []
    for product_id, group in monthly_sales.groupby("product_id"):
        ordered = group.sort_values("month_start").reset_index(drop=True)
        series = ordered["monthly_sales"].astype(float)
        overall_mean = float(series.mean()) if not series.empty else 0.0
        monthly_profile = ordered.assign(month_number=ordered["month_start"].dt.month).groupby("month_number")["monthly_sales"].mean()
        if overall_mean > 0:
            seasonality_strength = float((monthly_profile - overall_mean).abs().mean() / overall_mean)
        else:
            seasonality_strength = 0.0
        trend_strength = _trend_strength(series) if overall_mean > 0 else 0.0
        stats.append(
            {
                "product_id": product_id,
                "base_price_derived": round(float(ordered["average_price"].median()), 2),
                "seasonality_strength": round(_clamp(seasonality_strength, 0.0, 1.0), 3),
                "trend_strength": round(_clamp(trend_strength, -1.0, 1.0), 4),
            }
        )

    stats_frame = pd.DataFrame(stats)
    products = products.merge(stats_frame, on="product_id", how="inner")
    products["base_price"] = products["base_price_derived"].fillna(products["base_price"]).fillna(0.0).round(2)
    products["seasonality_strength"] = products["seasonality_strength"].fillna(0.0)
    products["trend_strength"] = products["trend_strength"].fillna(0.0)
    products["category"] = products["category"].replace("", "General Merchandise").fillna("General Merchandise")
    return products[
        ["product_id", "product_name", "category", "base_price", "seasonality_strength", "trend_strength"]
    ].sort_values("product_id").reset_index(drop=True)


def build_inventory_table(monthly_sales: pd.DataFrame, inventory_snapshot: pd.DataFrame) -> pd.DataFrame:
    inventory_base = (
        inventory_snapshot.groupby("product_id", as_index=False)
        .agg(
            stock_on_hand=("stock_on_hand", "sum"),
            reorder_point=("reorder_point", "sum"),
            safety_stock=("safety_stock", "sum"),
        )
    )
    inventory = monthly_sales.merge(inventory_base, on="product_id", how="left")
    prior_mean = (
        inventory.groupby("product_id")["monthly_sales"]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    demand_proxy = prior_mean.fillna(inventory.groupby("product_id")["monthly_sales"].shift(1)).fillna(inventory["monthly_sales"])
    static_buffer = inventory["stock_on_hand"].fillna(0.0) + inventory["safety_stock"].fillna(0.0) + inventory["reorder_point"].fillna(0.0)
    inventory["inventory_units"] = (static_buffer.combine(demand_proxy * 1.25, max).apply(math.ceil)).astype(int)
    return inventory[["product_id", "month_start", "inventory_units"]].sort_values(["product_id", "month_start"]).reset_index(drop=True)


def build_promotions_table(sales: pd.DataFrame, promotions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    promo_map = promotions.copy()
    sales_with_promos = sales.copy()
    sales_with_promos["month_start"] = sales_with_promos["date"].dt.to_period("M").dt.to_timestamp()
    sales_with_promos["observed_discount_pct"] = pd.to_numeric(sales_with_promos.get("discount_pct", 0.0), errors="coerce").fillna(0.0)

    promo_sales = sales_with_promos[sales_with_promos["promo_id"].ne("")].merge(
        promo_map[["promo_id", "discount_pct"]] if "discount_pct" in promo_map.columns else promo_map[["promo_id"]],
        on="promo_id",
        how="left",
        suffixes=("_sales", "_promo"),
    )
    promo_discount_pct = pd.to_numeric(promo_sales.get("discount_pct_promo", 0.0), errors="coerce").fillna(0.0)
    observed_discount_pct = pd.to_numeric(promo_sales.get("observed_discount_pct", 0.0), errors="coerce").fillna(0.0)
    promo_sales["promo_discount"] = pd.concat([observed_discount_pct, promo_discount_pct], axis=1).max(axis=1) / 100.0

    targeted_product_promos = promo_map[promo_map["target_type"].isin(["sku", "product", "item"]) & promo_map["target_value"].ne("")].copy()
    targeted_product_promos = targeted_product_promos.rename(columns={"target_value": "product_id"})
    targeted_product_promos["product_id"] = targeted_product_promos["product_id"].astype(str).str.strip()

    targeted_category_promos = promo_map[promo_map["target_type"].isin(["category", "department"]) & promo_map["target_value"].ne("")].copy()
    targeted_category_promos["target_value"] = targeted_category_promos["target_value"].astype(str).str.strip()
    category_matches = targeted_category_promos.merge(
        products[["product_id", "category"]],
        left_on="target_value",
        right_on="category",
        how="inner",
    )

    expanded = pd.concat(
        [
            targeted_product_promos[["promo_id", "product_id", "start_date", "end_date", "discount_pct"]],
            category_matches[["promo_id", "product_id", "start_date", "end_date", "discount_pct"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    expanded["month_start"] = expanded["start_date"].dt.to_period("M").dt.to_timestamp()
    expanded_promos = expanded.groupby(["product_id", "month_start"], as_index=False).agg(promo_discount=("discount_pct", "max"))
    expanded_promos["promo_discount"] = expanded_promos["promo_discount"] / 100.0

    if promo_sales.empty:
        observed_promos = sales_with_promos[sales_with_promos["observed_discount_pct"].gt(0)].groupby(
            ["product_id", "month_start"], as_index=False
        ).agg(observed_discount=("observed_discount_pct", lambda values: float(pd.Series(values).max()) / 100.0))
    else:
        observed_promos = promo_sales.groupby(["product_id", "month_start"], as_index=False).agg(observed_discount=("promo_discount", "max"))

    monthly = expanded_promos.merge(observed_promos, on=["product_id", "month_start"], how="outer")
    monthly["promo_intensity"] = monthly["observed_discount"].fillna(monthly["promo_discount"]).fillna(0.0).clip(lower=0.0, upper=0.8)
    monthly["supplier_delay_flag"] = 0
    return monthly[["product_id", "month_start", "promo_intensity", "supplier_delay_flag"]].sort_values(
        ["product_id", "month_start"]
    ).reset_index(drop=True)


def transform_kaggle_retail_to_raw_tables(source_tables: dict[str, pd.DataFrame], params: dict[str, Any]) -> dict[str, pd.DataFrame]:
    dataset_cfg = params["dataset"]
    product_limit = int(dataset_cfg.get("product_limit", 150))
    min_history_months = int(params["training"].get("min_history_months", 6))

    sales = clean_kaggle_sales(source_tables["sales"])
    products = clean_kaggle_products(source_tables["products"])
    inventory_snapshot = clean_kaggle_inventory(source_tables["inventory"])
    promotions = clean_kaggle_promotions(source_tables["promotions"])

    sales = sales.merge(products[["product_id", "product_name", "category"]], on="product_id", how="inner")
    monthly_sales = build_monthly_sales_frame(sales)

    ranked_products = (
        monthly_sales.groupby("product_id", as_index=False)
        .agg(active_months=("month_start", "nunique"), total_units=("monthly_sales", "sum"))
        .query("active_months >= @min_history_months")
        .sort_values(["active_months", "total_units"], ascending=[False, False])
    )
    selected_product_ids = ranked_products["product_id"] if product_limit <= 0 else ranked_products["product_id"].head(product_limit)
    if selected_product_ids.empty:
        raise ValueError("No Kaggle products met the minimum history requirement.")

    sales = sales[sales["product_id"].isin(selected_product_ids)].copy()
    monthly_sales = monthly_sales[monthly_sales["product_id"].isin(selected_product_ids)].copy()
    products = build_products_table(products[products["product_id"].isin(selected_product_ids)].copy(), monthly_sales)
    inventory = build_inventory_table(monthly_sales, inventory_snapshot)
    monthly_promotions = build_promotions_table(sales, promotions, products)

    daily_sales = (
        sales.groupby(["date", "product_id"], as_index=False)
        .agg(quantity=("quantity", "sum"), sales_value=("sales_value", "sum"))
    )
    daily_sales["unit_price"] = daily_sales["sales_value"] / daily_sales["quantity"]
    daily_sales = daily_sales[["date", "product_id", "quantity", "unit_price"]].sort_values(["product_id", "date"]).reset_index(drop=True)

    return {
        "products": products,
        "sales": daily_sales,
        "inventory": inventory,
        "promotions": monthly_promotions,
    }


def load_raw_data(paths: RawPaths) -> dict[str, pd.DataFrame]:
    return {
        "products": pd.read_csv(paths.products),
        "sales": pd.read_csv(paths.sales, parse_dates=["date"]),
        "inventory": pd.read_csv(paths.inventory, parse_dates=["month_start"]),
        "promotions": pd.read_csv(paths.promotions, parse_dates=["month_start"]),
    }


def build_request_rows(
    params: dict[str, Any],
    item_count: int,
    tax_rate: float,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    dataset_cfg = params.setdefault("dataset", {})
    dataset_cfg.setdefault("kaggle_dataset", DEFAULT_KAGGLE_DATASET)
    dataset_cfg.setdefault("source_extract_dir", str(DEFAULT_SOURCE_DIR))
    dataset_version = dataset_cfg.get("dataset_version", "kaggle-synthetic-retail")

    raw_paths = ensure_dataset(
        {
            "paths": {"raw_dir": str(DEFAULT_RAW_DIR)},
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
    products["product_id"] = products["product_id"].astype(str).str.strip()
    inventory["product_id"] = inventory["product_id"].astype(str).str.strip()
    monthly_promotions["product_id"] = monthly_promotions["product_id"].astype(str).str.strip()

    source_dir = Path(dataset_cfg["source_extract_dir"])
    resolved_source_dir = source_dir if source_dir.is_absolute() else raw_paths.products.parent / source_dir
    source_tables = load_kaggle_source_tables(resolved_source_dir)
    sales = clean_kaggle_sales(source_tables["sales"])
    monthly_sales = build_monthly_sales_frame(sales)
    monthly_sales["product_id"] = monthly_sales["product_id"].astype(str).str.strip()

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
    available_product_ids = set(products["product_id"].astype(str))
    eligible = eligible[eligible["product_id"].astype(str).isin(available_product_ids)].copy()
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
    params: dict[str, Any],
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
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory where the Kaggle dataset will be extracted or reused.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT),
        help="Directory where the generated file will be written.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "json"),
        default="csv",
        help="Write flattened forecast rows as CSV or MCP request payloads as JSON.",
    )
    args = parser.parse_args(argv)

    params = default_params(args.source_dir)

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
