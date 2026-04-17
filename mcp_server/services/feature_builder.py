from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any


def _stable_code(value: str, modulus: int = 10_000) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulus


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _rolling(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


@dataclass
class BuiltFeatures:
    ordered_names: list[str]
    ordered_values: list[float]
    raw_features: dict[str, float]


def build_features(request: dict[str, Any], manifest: dict[str, Any]) -> BuiltFeatures:
    product_context = request.get("product_context", {})
    history = request.get("history", {})
    current = request.get("current_context", {})
    target_month = datetime.fromisoformat(request["target_month"])

    lags = [
        float(history.get(f"monthly_sales_lag_{index}", 0.0))
        for index in range(1, 7)
    ]
    roll_3 = _rolling(lags[:3])
    roll_6 = _rolling(lags)

    lag_1 = lags[0] if len(lags) > 0 else 0.0
    lag_2 = lags[1] if len(lags) > 1 else 0.0
    lag_3 = lags[2] if len(lags) > 2 else 0.0

    average_price = float(current.get("average_price", product_context.get("base_price", 0.0)))
    inventory_units = float(current.get("inventory_units", 0.0))
    promo_intensity = float(current.get("promo_intensity", 0.0))
    trend_strength = float(product_context.get("trend_strength", 0.0))
    base_price = float(product_context.get("base_price", average_price))

    raw_features: dict[str, float] = {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_4": lags[3] if len(lags) > 3 else 0.0,
        "lag_5": lags[4] if len(lags) > 4 else 0.0,
        "lag_6": lags[5] if len(lags) > 5 else 0.0,
        "monthly_sales_lag_1": lag_1,
        "monthly_sales_lag_2": lag_2,
        "monthly_sales_lag_3": lag_3,
        "monthly_sales_lag_4": lags[3] if len(lags) > 3 else 0.0,
        "monthly_sales_lag_5": lags[4] if len(lags) > 4 else 0.0,
        "monthly_sales_lag_6": lags[5] if len(lags) > 5 else 0.0,
        "rolling_mean_3": roll_3["mean"],
        "rolling_std_3": roll_3["std"],
        "rolling_min_3": roll_3["min"],
        "rolling_max_3": roll_3["max"],
        "rolling_mean_6": roll_6["mean"],
        "rolling_std_6": roll_6["std"],
        "rolling_min_6": roll_6["min"],
        "rolling_max_6": roll_6["max"],
        "diff_1": lag_1 - lag_2,
        "diff_2": lag_2 - lag_3,
        "mom_ratio_1": _safe_ratio(lag_1, lag_2),
        "mom_ratio_2": _safe_ratio(lag_2, lag_3),
        "lag_1_share_of_6m": _safe_ratio(lag_1, sum(lags)),
        "month_number": float(target_month.month),
        "quarter": float(((target_month.month - 1) // 3) + 1),
        "month_sin": math.sin(2 * math.pi * target_month.month / 12),
        "month_cos": math.cos(2 * math.pi * target_month.month / 12),
        "product_code": float(_stable_code(str(request.get("product_id", "")))),
        "category_code": float(_stable_code(str(product_context.get("category", "")))),
        "average_price": average_price,
        "base_price": base_price,
        "sales_value": lag_1 * average_price,
        "active_days": float(request.get("current_context", {}).get("active_days", 30.0)),
        "daily_qty_std": roll_3["std"],
        "daily_qty_max": max(lags) if lags else 0.0,
        "daily_qty_min": min(lags) if lags else 0.0,
        "price_std": 0.0,
        "price_min": average_price,
        "price_max": average_price,
        "weekend_units": 0.0,
        "month_end_units": 0.0,
        "active_store_count": 1.0,
        "top_store_units": lag_1,
        "store_sales_std": 0.0,
        "days_in_month": float(target_month.day if target_month.day > 1 else 30),
        "avg_daily_units": _safe_ratio(lag_1, float(request.get("current_context", {}).get("active_days", 30.0))),
        "avg_daily_sales_value": _safe_ratio(lag_1 * average_price, float(request.get("current_context", {}).get("active_days", 30.0))),
        "weekend_share": 0.0,
        "month_end_share": 0.0,
        "price_range": 0.0,
        "price_cv": 0.0,
        "active_day_ratio": _safe_ratio(float(request.get("current_context", {}).get("active_days", 30.0)), 30.0),
        "observed_days_in_month": float(request.get("current_context", {}).get("active_days", 30.0)),
        "month_completeness_ratio": _safe_ratio(float(request.get("current_context", {}).get("active_days", 30.0)), 30.0),
        "is_complete_month": 1.0,
        "top_store_share": 1.0,
        "inventory_units": inventory_units,
        "promo_intensity": promo_intensity,
        "supplier_delay_flag": float(current.get("supplier_delay_flag", 0.0)),
        "promo_day_count": float(current.get("promo_day_count", 0.0)),
        "promo_event_count": float(current.get("promo_event_count", 0.0)),
        "trend_strength": trend_strength,
        "seasonality_strength": float(product_context.get("seasonality_strength", 0.0)),
        "price_to_base_ratio": _safe_ratio(average_price, base_price),
        "stock_coverage_months": _safe_ratio(inventory_units, max(lag_1, 1.0)),
        "stock_coverage_rolling": _safe_ratio(inventory_units, max(roll_3["mean"], 1.0)),
        "promo_x_lag1": promo_intensity * lag_1,
        "promo_x_inventory": promo_intensity * inventory_units,
        "price_x_trend": _safe_ratio(average_price, base_price) * trend_strength,
        "demand_to_value_ratio": _safe_ratio(lag_1, lag_1 * average_price),
    }

    ordered_names = manifest.get("ordered_feature_names", [])
    if not ordered_names:
        raise ValueError("feature manifest is missing ordered_feature_names")

    ordered_values = [float(raw_features.get(name, 0.0)) for name in ordered_names]
    return BuiltFeatures(ordered_names=ordered_names, ordered_values=ordered_values, raw_features=raw_features)
