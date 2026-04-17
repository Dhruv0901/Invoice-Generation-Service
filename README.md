# Invoice Generation Service

This repo forecasts monthly product demand from a synthetic Kaggle retail dataset and generates pro forma invoice `.docx` files through the MCP server.

## Current Flow

1. `scripts/generate_next_month_forecast.py` ensures the Kaggle dataset is available.
2. The script samples 30 eligible products with at least 6 months of history.
3. Each sampled row is converted into an MCP-compatible invoice request.
4. `mcp_server` forecasts demand and writes invoice documents plus metadata.
5. If a Qwen 2.5 endpoint is configured, Qwen drafts the invoice text; otherwise the renderer falls back to deterministic text.

## Credentials

Fill in these files:

- `config/kaggle_credentials.json`
- `config/qwen_config.json`

You can also use environment variables from `.env.example`.

## Run

Generate 30 MCP request payloads:

```powershell
.\.venv\Scripts\python scripts\generate_next_month_forecast.py --count 30 --output-format json
```

Generate invoices from the MCP server:

```powershell
.\.venv\Scripts\python -m mcp_server.server generate_invoices_from_kaggle_sample
```

## Dataset Options

The Kaggle ingestion layer accepts any of the following:

- extracted CSVs already present in `forecast_core/data/raw/kaggle-synthetic-retail/`
- a Kaggle zip file already present in that folder
- Kaggle credentials in `config/kaggle_credentials.json`, which allows the repo to download the archive directly

## Structure

- `forecast_core/`: dataset prep and training assets
- `mcp_server/`: MCP entrypoints, forecasting, invoice generation
- `scripts/`: request generation utilities
- `config/`: local credential and provider config
- `tests/`: focused regression tests
