from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InvoicePricing:
    billed_quantity: int
    unit_price: float
    line_total: float
    subtotal: float
    tax: float
    grand_total: float


def select_unit_price(request: dict[str, Any]) -> float:
    if request.get("unit_price") is not None:
        return float(request["unit_price"])

    current = request.get("current_context", {})
    product = request.get("product_context", {})

    if current.get("average_price") is not None:
        return float(current["average_price"])
    return float(product.get("base_price", 0.0))


def build_invoice_pricing(
    request: dict[str, Any],
    predicted_monthly_sales: float,
    tax_rate: float,
) -> InvoicePricing:
    billed_quantity = max(0, round(predicted_monthly_sales))
    unit_price = select_unit_price(request)
    line_total = round(billed_quantity * unit_price, 2)
    subtotal = line_total
    tax = round(subtotal * tax_rate, 2)
    grand_total = round(subtotal + tax, 2)
    return InvoicePricing(
        billed_quantity=billed_quantity,
        unit_price=unit_price,
        line_total=line_total,
        subtotal=subtotal,
        tax=tax,
        grand_total=grand_total,
    )

