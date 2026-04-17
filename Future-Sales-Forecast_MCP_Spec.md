# Functional Specification: Future-Sales-Forecast as an MCP Forecasting and Invoice Generation Service


## 1. Purpose

Build a production-oriented service around the `Dhruv0901/Future-Sales-Forecast` repository so that an MCP client can:

1. send structured sales history and commercial context,
2. obtain a demand prediction from the trained model in this repo,
3. transform that prediction into an editable invoice in `.docx` format,
4. return both machine-readable forecast output and a downloadable invoice artifact.

The implementation must keep the existing repository as the forecasting core and expose it through an MCP server rather than rewriting the modeling logic from scratch.

---

## 2. Repo baseline and design implications

The current repository is already an end-to-end training pipeline, but it is **not yet a serving system**.

### Confirmed repo characteristics
- The repo contains a DVC pipeline with four stages: `data_ingestion`, `feature_selection`, `model_training`, and `model_evaluation`.
- `data_ingestion` writes `data/raw/train.csv` and `data/raw/test.csv`.
- `feature_selection` writes processed training/test files under `data/processed`.
- `model_training` saves the trained model to `models/model.pkl`.
- `model_evaluation` writes metrics to `reports/metrics.json`.
- The modeling stack uses `scikit-learn`, `xgboost`, `mlflow`, `dvc`, `numpy`, `pandas`, and `PyYAML`.
- The current model is a `Pipeline(StandardScaler -> TransformedTargetRegressor(XGBRegressor))`.
- The pipeline engineers product-month forecasting features including lags 1-6, rolling aggregates, inventory, promotions, seasonality, category/product encodings, and optional external signals.
- The split strategy in `params.yaml` is currently `time_based` with `time_holdout_months: 2`.
- The target variable is `monthly_sales`.

### Important implementation consequence
Because the training flow selects columns for `data/processed/train.csv` and the model is trained by positional column order, the serving layer **must persist and reuse the exact feature manifest and order** used at train time. This is the single most important change required for reliable inference.

---

## 3. Solution overview

### High-level architecture
Use the existing repo as the forecasting engine, then add two thin layers:

1. **MCP server layer**  
   Exposes forecast and invoice generation as MCP tools/resources/prompts.

2. **Document generation layer**  
   Uses an open-source API to merge forecast output into an editable invoice template.

### Chosen open-source API
Use **Carbone On-Premise** as the document-generation API.

### Why Carbone is the best fit here
- It is purpose-built for generating documents from JSON plus templates.
- It supports `DOCX` as both a template format and an output format.
- It exposes REST endpoints for template upload and report generation.
- It is self-hostable/on-premise, which keeps the whole stack compatible with an open-source, infrastructure-controlled deployment model.
- It is especially suitable for invoice templating because the invoice layout can be designed in Word/LibreOffice while the service only injects data.

### Why not make the model itself generate the invoice
The forecasting repo should stay responsible for **prediction**. Document composition is a separate concern and should be isolated behind a document API so templates can evolve without retraining or refactoring the ML pipeline.

---

## 4. In-scope deliverable

Build a system that:

- loads the trained model from this repo,
- validates incoming sales/forecast requests,
- computes the feature vector expected by the trained model,
- predicts next-period `monthly_sales`,
- derives invoice line items from the prediction,
- renders an editable `.docx` invoice,
- returns metadata that allows an MCP client to display the result or fetch the file.

Out of scope for v1:
- online model retraining triggered from MCP,
- multi-tenant auth/permissions,
- in-browser collaborative document editing,
- payment collection,
- ERP synchronization beyond a simple webhook/export stub.

---

## 5. Target operating model

### Forecasting mode
The repo currently forecasts at the **product-month** level.  
Therefore the serving layer should predict **next month demand** for a given product or small batch of products, not next-minute or store-basket demand.

### Invoice mode
The invoice should represent a **forecast-backed pro forma invoice / planned replenishment invoice** rather than a confirmed order invoice.  
This distinction must be explicit in the document footer and metadata.

Recommended document label:
**“Forecast-Based Pro Forma Invoice”**

This avoids misleading downstream users into thinking the document represents an already fulfilled sale.

---

## 6. End-to-end flow

1. MCP client calls `forecast_and_generate_invoice`.
2. MCP server validates the request payload.
3. MCP server transforms the payload into the same engineered feature schema expected by the model.
4. MCP server loads `models/model.pkl`.
5. MCP server predicts `monthly_sales`.
6. Business rules convert predicted demand into invoice quantity and monetary values.
7. MCP server sends JSON data plus template ID to Carbone.
8. Carbone returns a generated editable `DOCX`.
9. MCP server stores the file locally or in object storage.
10. MCP server returns:
   - prediction values,
   - invoice metadata,
   - file path / signed URL / storage key,
   - trace information such as model version and template version.

---

## 7. Required repository changes

Create the following new folders/files inside the repo:

```text
Future-Sales-Forecast/
├── mcp_server/
│   ├── server.py
│   ├── tools/
│   │   ├── forecast.py
│   │   ├── invoice.py
│   │   ├── health.py
│   │   └── metadata.py
│   ├── schemas/
│   │   ├── forecast_request.schema.json
│   │   ├── forecast_response.schema.json
│   │   ├── invoice_request.schema.json
│   │   └── invoice_response.schema.json
│   ├── services/
│   │   ├── model_loader.py
│   │   ├── feature_builder.py
│   │   ├── pricing.py
│   │   ├── carbone_client.py
│   │   ├── storage.py
│   │   └── audit.py
│   └── resources/
│       └── sample_payloads/
├── templates/
│   └── invoice_template.docx
├── artifacts/
│   └── invoices/
├── models/
│   ├── model.pkl
│   └── feature_manifest.json
├── reports/
│   └── metrics.json
└── .env.example
```

### Mandatory training-time enhancement
Update the training or feature-selection stage so it also writes:

```json
models/feature_manifest.json
```

This file must contain:
- ordered feature names,
- target variable name,
- model training timestamp,
- model type,
- data split strategy,
- repo commit SHA if available,
- params snapshot.

Without this file, inference will be fragile.

---

## 8. MCP server contract

### Server name
`future-sales-forecast-mcp`

### Preferred transport
- **Primary**: stdio for local Codex/client integration
- **Optional**: Streamable HTTP for remote deployment

### MCP tools

#### 8.1 `health_check`
Returns service health, model availability, template availability, and Carbone connectivity.

**Input**
```json
{}
```

**Output**
```json
{
  "status": "ok",
  "model_loaded": true,
  "feature_manifest_loaded": true,
  "carbone_reachable": true,
  "template_ready": true
}
```

#### 8.2 `get_model_metadata`
Returns model and dataset metadata for the loaded forecasting artifact.

**Output**
```json
{
  "model_name": "xgboost_monthly_sales_forecaster",
  "target_variable": "monthly_sales",
  "feature_count": 48,
  "training_split_strategy": "time_based",
  "holdout_months": 2,
  "metrics": {
    "Val RMSE": 387.30,
    "Val R2": 0.9074,
    "Val MAE": 47.74,
    "Val MAPE": 0.0179
  }
}
```

#### 8.3 `forecast_sales`
Predicts next-period monthly sales without generating an invoice.

**Input**
```json
{
  "request_id": "fcst_001",
  "target_month": "2026-05-01",
  "product_id": "SKU-10045",
  "product_context": {
    "product_name": "Wireless Mouse",
    "category": "Accessories",
    "base_price": 29.99,
    "trend_strength": 0.08,
    "seasonality_strength": 0.21
  },
  "history": {
    "monthly_sales_lag_1": 420,
    "monthly_sales_lag_2": 390,
    "monthly_sales_lag_3": 405,
    "monthly_sales_lag_4": 376,
    "monthly_sales_lag_5": 398,
    "monthly_sales_lag_6": 365
  },
  "current_context": {
    "average_price": 28.50,
    "inventory_units": 600,
    "promo_intensity": 0.20,
    "supplier_delay_flag": 0,
    "promo_day_count": 5,
    "promo_event_count": 1
  },
  "external_signals": {
    "macro_index": 1.02,
    "holiday_flag": 0
  }
}
```

**Output**
```json
{
  "request_id": "fcst_001",
  "target_month": "2026-05-01",
  "product_id": "SKU-10045",
  "predicted_monthly_sales": 447.82,
  "rounded_planned_quantity": 448,
  "model_version": "model.pkl",
  "feature_manifest_version": "feature_manifest.json",
  "assumptions": [
    "prediction generated for next monthly period",
    "invoice quantity rounded to nearest whole unit"
  ]
}
```

#### 8.4 `generate_invoice_docx`
Generates an editable invoice from forecast output plus commercial data.

**Input**
```json
{
  "request_id": "inv_001",
  "customer": {
    "name": "Acme Retail Pty Ltd",
    "billing_address": "Level 10, 123 George St, Sydney NSW 2000",
    "email": "ap@acmeretail.com"
  },
  "seller": {
    "name": "Future Sales Forecast Demo Store",
    "abn": "11 111 111 111",
    "address": "1 Forecast Way, Sydney NSW 2000"
  },
  "currency": "AUD",
  "tax_rate": 0.10,
  "invoice_date": "2026-04-17",
  "target_month": "2026-05-01",
  "items": [
    {
      "product_id": "SKU-10045",
      "product_name": "Wireless Mouse",
      "unit_price": 28.50,
      "forecast_context": {
        "history": {
          "monthly_sales_lag_1": 420,
          "monthly_sales_lag_2": 390,
          "monthly_sales_lag_3": 405,
          "monthly_sales_lag_4": 376,
          "monthly_sales_lag_5": 398,
          "monthly_sales_lag_6": 365
        },
        "product_context": {
          "category": "Accessories",
          "base_price": 29.99,
          "trend_strength": 0.08,
          "seasonality_strength": 0.21
        },
        "current_context": {
          "average_price": 28.50,
          "inventory_units": 600,
          "promo_intensity": 0.20,
          "supplier_delay_flag": 0,
          "promo_day_count": 5,
          "promo_event_count": 1
        },
        "external_signals": {
          "macro_index": 1.02,
          "holiday_flag": 0
        }
      }
    }
  ]
}
```

**Output**
```json
{
  "request_id": "inv_001",
  "invoice_number": "PFI-20260417-0001",
  "document_type": "forecast_based_pro_forma_invoice",
  "target_month": "2026-05-01",
  "line_items": [
    {
      "product_id": "SKU-10045",
      "product_name": "Wireless Mouse",
      "predicted_quantity": 447.82,
      "billed_quantity": 448,
      "unit_price": 28.50,
      "line_total": 12768.00
    }
  ],
  "subtotal": 12768.00,
  "tax": 1276.80,
  "grand_total": 14044.80,
  "docx_path": "artifacts/invoices/PFI-20260417-0001.docx",
  "model_trace": {
    "model_version": "model.pkl",
    "feature_manifest_version": "feature_manifest.json",
    "template_version": "invoice_template_v1"
  }
}
```

#### 8.5 `forecast_and_generate_invoice`
Convenience tool that runs the full pipeline in a single call.

This should be the default tool used by Codex or other MCP clients.

---

## 9. MCP resources

Expose these read-only resources:

### `forecast://metrics/latest`
Returns current evaluation metrics loaded from `reports/metrics.json`.

### `forecast://model/manifest`
Returns `models/feature_manifest.json`.

### `forecast://template/invoice-schema`
Returns the supported JSON payload structure for invoice rendering.

### `forecast://examples/minimal-request`
Returns a minimal valid example for one-product forecasting.

These resources help LLM clients ground themselves before calling tools.

---

## 10. Forecast input design

### Recommended serving contract
Do **not** require raw daily transactions for v1 inference.  
Instead accept **already summarized monthly context**, because the trained model ultimately consumes engineered monthly features.

### Minimum required history
Because the feature engineering uses `lag_1` through `lag_6`, the request must contain **at least six complete months of sales history** for each product.

### Required fields per product
- `product_id`
- `product_name`
- `category`
- `base_price`
- `trend_strength`
- `seasonality_strength`
- six monthly sales lags
- `average_price`
- `inventory_units`
- `promo_intensity`
- `supplier_delay_flag`
- `promo_day_count`
- `promo_event_count`
- optional external signals matching the training feature manifest

### Server-side derived features
The service must compute, at minimum:
- `rolling_mean_3`
- `rolling_std_3`
- `rolling_min_3`
- `rolling_max_3`
- `rolling_mean_6`
- `rolling_std_6`
- `rolling_min_6`
- `rolling_max_6`
- `diff_1`
- `diff_2`
- `mom_ratio_1`
- `mom_ratio_2`
- `lag_1_share_of_6m`
- `month_number`
- `quarter`
- `month_sin`
- `month_cos`
- `product_code`
- `category_code`
- `price_to_base_ratio`
- `stock_coverage_months`
- `stock_coverage_rolling`
- `promo_x_lag1`
- `promo_x_inventory`
- `price_x_trend`
- any other fields required by `feature_manifest.json`

### Strict rule
At inference time, the final dataframe passed into `model.predict()` must be reordered to **exactly** match `feature_manifest.json`.

---

## 11. Inference business rules

### Quantity rounding
Predicted quantity is continuous because it comes from a regression model.  
Invoice quantity must be integer.

Default strategy:
- `billed_quantity = max(0, round(predicted_monthly_sales))`

Optional future strategies:
- ceiling for conservative stocking,
- floor for conservative invoicing,
- safety-stock uplift multiplier.

### Pricing
Default unit price priority:
1. `request.items[n].unit_price`
2. `current_context.average_price`
3. `product_context.base_price`

### Tax
Use request-level `tax_rate`.

### Monetary calculations
- `line_total = billed_quantity * unit_price`
- `subtotal = sum(line_total)`
- `tax = subtotal * tax_rate`
- `grand_total = subtotal + tax`

### Document watermark / disclaimer
The invoice must contain a visible note:

> This is a forecast-based pro forma invoice generated from a machine learning demand prediction and is not a tax invoice for fulfilled goods unless separately approved.

---

## 12. Carbone integration design

### Deployment style
Run Carbone as a separate container/service in the same deployment stack.

### API usage pattern
1. Upload template once using `/template`.
2. Store returned template ID in config.
3. For each invoice request:
   - send merged JSON to `/render/{templateId}`
   - request `docx` output
   - download or directly stream the generated file

### Template strategy
Store a Word template at:
```text
templates/invoice_template.docx
```

The template should contain placeholders for:
- seller block
- customer block
- invoice number
- invoice date
- target month
- line items table
- subtotal
- tax
- grand total
- forecast disclaimer
- model trace block

### Carbone payload example
```json
{
  "invoice_number": "PFI-20260417-0001",
  "invoice_date": "2026-04-17",
  "target_month": "May 2026",
  "seller": {
    "name": "Future Sales Forecast Demo Store",
    "abn": "11 111 111 111",
    "address": "1 Forecast Way, Sydney NSW 2000"
  },
  "customer": {
    "name": "Acme Retail Pty Ltd",
    "billing_address": "Level 10, 123 George St, Sydney NSW 2000",
    "email": "ap@acmeretail.com"
  },
  "items": [
    {
      "product_id": "SKU-10045",
      "product_name": "Wireless Mouse",
      "quantity": 448,
      "unit_price": 28.50,
      "line_total": 12768.00
    }
  ],
  "subtotal": 12768.00,
  "tax": 1276.80,
  "grand_total": 14044.80,
  "document_note": "Forecast-based pro forma invoice",
  "model_trace": {
    "model_version": "model.pkl",
    "template_version": "invoice_template_v1"
  }
}
```

---

## 13. Editable DOCX requirement

The generated invoice must remain editable after generation.

### Mandatory rules
- Output format must be `.docx`, not PDF-only.
- The generated file must not flatten text into images.
- The line item table must remain editable in Word/LibreOffice.
- Headers, addresses, totals, and disclaimer must remain standard editable text.
- If a PDF preview is needed later, create it as a secondary export, not the primary artifact.

---

## 14. Storage and artifact handling

### Local-first v1
For v1, store generated documents under:
```text
artifacts/invoices/
```

### Filename convention
```text
PFI-YYYYMMDD-XXXX.docx
```

### Metadata sidecar
Write a JSON sidecar next to each DOCX:
```text
PFI-YYYYMMDD-XXXX.json
```

This sidecar should contain:
- request payload hash,
- prediction values,
- invoice totals,
- model version,
- template version,
- generation timestamp,
- storage path.

This makes the system auditable.

---

## 15. Error handling

### Validation errors
Return structured errors when:
- fewer than 6 months of history are supplied,
- required product fields are missing,
- invoice customer/seller info is incomplete,
- a required external signal from the manifest is absent.

### Model errors
Return structured errors when:
- `model.pkl` missing,
- `feature_manifest.json` missing,
- feature vector shape mismatch,
- prediction returns NaN or inf.

### Document errors
Return structured errors when:
- Carbone is unavailable,
- template ID invalid,
- DOCX render fails,
- storage write fails.

Error example:
```json
{
  "error": {
    "code": "FEATURE_MANIFEST_MISMATCH",
    "message": "Incoming payload could not be transformed into the exact feature schema expected by the model.",
    "details": {
      "missing_features": ["rolling_mean_6", "promo_x_inventory"]
    }
  }
}
```

---

## 16. Observability and audit

### Logging
Each forecast/invoice call must log:
- request ID,
- invoice number,
- product IDs,
- prediction values,
- model version,
- template version,
- generation latency,
- outcome status.

### Metrics
Track:
- forecast request count,
- invoice generation success rate,
- average prediction latency,
- average document generation latency,
- Carbone error count.

### Audit trail
Every invoice response should include:
- `request_id`
- `invoice_number`
- `model_trace`
- `generated_at`

---

## 17. Security and safety controls

### Input safety
- enforce JSON schema validation before feature building,
- cap max item count in a single request,
- sanitize user-provided text before passing into template data,
- restrict template selection to allowlisted IDs.

### File safety
- never allow arbitrary user upload as an executable template in v1,
- store templates in repo or controlled object storage,
- do not expose local filesystem paths outside trusted environments.

### Forecast interpretation safety
Always label the output as forecast-based and include disclaimers so users do not mistake it for a confirmed order or statutory tax invoice.

---

## 18. Recommended implementation sequence for Codex

### Phase 1: Make inference deterministic
1. add `feature_manifest.json` generation,
2. implement `model_loader.py`,
3. implement `feature_builder.py`,
4. verify one manual inference path works.

### Phase 2: Add MCP surface
5. implement `health_check`,
6. implement `get_model_metadata`,
7. implement `forecast_sales`,
8. expose sample resources.

### Phase 3: Add invoice rendering
9. create `templates/invoice_template.docx`,
10. implement `carbone_client.py`,
11. implement `pricing.py`,
12. implement `generate_invoice_docx`.

### Phase 4: Full orchestration
13. implement `forecast_and_generate_invoice`,
14. save artifacts + sidecars,
15. add integration tests.

---

## 19. Acceptance criteria

The implementation is complete when all of the following are true:

1. An MCP client can call one tool with forecast context and receive a prediction.
2. The service loads the trained model from this repo instead of calling an external ML API.
3. The service uses a persisted feature manifest to guarantee inference column order.
4. The service can generate a `.docx` invoice from the forecast result.
5. The invoice is editable in Word/LibreOffice.
6. The invoice clearly states it is forecast-based / pro forma.
7. A JSON sidecar audit file is written for each invoice.
8. Health and metadata tools work even when no invoice is generated.
9. Errors are returned as structured JSON.
10. The system can run locally via Docker Compose.

---

## 20. Docker Compose target

A minimal local deployment should include:

- `forecast-mcp` service
- `carbone` service
- optional `minio` or simple local volume for artifacts

Example service intent:
```yaml
services:
  forecast-mcp:
    build: .
    env_file: .env
    depends_on:
      - carbone

  carbone:
    image: carbone/carbone-ee
```

If using a community/on-prem image or binary instead, adjust the image/source accordingly during implementation.

---

## 21. Test plan

### Unit tests
- feature derivation from 6-month history,
- exact feature ordering against manifest,
- rounding and price selection logic,
- tax and totals math.

### Integration tests
- forecast-only tool call,
- invoice-only render from synthetic forecast output,
- end-to-end MCP call producing a DOCX.

### Manual tests
- open generated DOCX in Microsoft Word,
- confirm line items are editable,
- confirm disclaimer and totals are present,
- confirm repeated requests produce deterministic output for identical input.

---

## 22. Future extensions

- batch forecasts for multiple products,
- quote documents in addition to invoices,
- PDF preview export alongside DOCX,
- approval workflow before invoice release,
- confidence intervals and scenario bands,
- UI for editing generated DOCX in-browser with a document suite.

---

## 23. Final implementation directive for Codex

Implement the existing `Future-Sales-Forecast` repo as the forecasting core of an MCP server.  
Do not replace the model training pipeline.  
Add a deterministic inference layer that reproduces the trained feature order, expose forecast and invoice-generation tools through MCP, use Carbone On-Premise as the open-source document API, and generate editable `.docx` forecast-based pro forma invoices from the model output.

The highest-priority engineering requirement is: **persist and enforce exact training feature order at inference time**.
