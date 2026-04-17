# Kaggle-Aligned Validation Payload

This file accompanies `kaggle_validation_request.json`.

It is designed to let you generate a forecasted invoice using a payload that matches:

- the Kaggle dataset structure referenced by `forecast_core`
- the service request schema used by `mcp_server`
- the repo's product-month forecasting assumptions

Source dataset page:
- https://www.kaggle.com/datasets/amirkhanh/synthetic-retail-dataset-1-2m-transactions

Important limitation:
- This is not a literal extracted row from the Kaggle CSV files because those files are not present in this repo and Kaggle downloads require authenticated access.
- The payload is intentionally schema-aligned and realistic enough for invoice generation and manual validation flows.

Relevant Kaggle file structure from the dataset card:
- `sku_master.csv`: product details such as `sku_id`, `sku_name`, `category`, and pricing fields
- `inventory_snapshot.csv`: stock fields such as `sku_id` and `stock_on_hand`
- `promotions.csv`: promo windows and discount metadata
- `sales_transactions.csv`: transactional fields such as `date`, `store_id`, `sku_id`, `quantity`, `unit_price`, and `promo_id`

How to use:

```powershell
python -m mcp_server.server forecast_and_generate_invoice --input ".\mcp_server\resources\sample_payloads\kaggle_validation_request.json"
```
