# Invoice Generation Service

This repo forecasts monthly product demand from a Kaggle retail dataset and generates pro forma invoice `.docx` files through the MCP server.

## Current Flow

1. `scripts/generate_next_month_forecast.py` ensures the Kaggle dataset is available.
2. The script samples eligible products with at least 6 months of history.
3. Each sampled row is converted into an MCP-compatible invoice request.
4. `mcp_server` forecasts demand and writes invoice documents plus metadata.
5. Qwen drafts the invoice text through the configured Ollama/OpenAI-compatible endpoint.

## Credentials

Fill in these files:

- `config/kaggle_credentials.json`
- `config/qwen_config.json`

You can also use `KAGGLE_USERNAME`, `KAGGLE_KEY`, `QWEN_BASE_URL`, `QWEN_API_KEY`, and `QWEN_MODEL`.

## Run

Generate 30 MCP request payloads:

```powershell
.\.venv\Scripts\python scripts\generate_next_month_forecast.py --count 30 --output-format json
```

Generate invoices from the MCP server:

```powershell
.\.venv\Scripts\python -m mcp_server.server generate_invoices_from_kaggle_sample
```

## Structure

- `mcp_server/`: MCP entrypoints, forecasting, invoice generation
- `scripts/`: request generation utilities
- `data/`: cached prepared data and optional Kaggle extract
- `config/`: local credential and provider config
- `tests/`: focused regression tests
