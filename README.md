# Invoice Generation Service

An MCP-style invoice generation layer built around the forecasting work in [`Dhruv0901/Future-Sales-Forecast`](https://github.com/Dhruv0901/Future-Sales-Forecast).
The idea is to keep model training and heavier forecasting experimentation in the upstream repo, then use this repo as the thinner serving layer that:

- accepts a compact sales context
- builds the forecast feature row
- predicts next-period demand
- sends the invoice payload to Ollama through an OpenAI-compatible endpoint
- returns a generated invoice artifact plus metadata

Right now the service works on a small request payload and a sample forecast flow, but the intended direction is broader: plugging this into real time or near-real time demand data, grouping invoice generation by company rather than one invoice per item, and eventually running it as an actual request-serving server instead of a local command-driven workflow.

* * *

## Current Operational Metrics

The current metrics below are not model-training metrics like RMSE or R2.
They are MCP-server-style operational metrics captured from a local benchmark run on April 19, 2026 using:

- local sample payload: `mcp_server/resources/sample_payloads/forecast_request.json`
- local Ollama endpoint: `http://localhost:11434/v1`
- model id: `qwen2.5:7b`
- benchmark script: `scripts/benchmark_mcp_server.py`

Metric | Value
--- | ---
`health_check` success rate | `100%` over 3 runs
`health_check` p95 latency | `0.79 ms`
`forecast_sales` success rate | `100%` over 5 runs
`forecast_sales` p95 latency | `25.38 ms`
Invoice generation latency | `55.03 s` p50/p95 on the current local Ollama setup

These metrics matter more for this repo than classic offline forecast metrics because this repo is acting as a tool-serving layer. The main question here is not only whether the forecast is numerically reasonable, but whether the tool can respond reliably, write artifacts safely, and stay usable when a client calls it through MCP.

* * *

## Architecture

The service is intentionally split across two repos:

- `Future-Sales-Forecast`: training, dataset shaping, feature experimentation, and model-development logic
- this repo: thin serving layer for forecast requests, invoice generation, audit logging, and MCP-style tool execution

That separation keeps the runtime smaller and easier to reason about, but it also means the serving layer depends on forecast artifacts and assumptions produced elsewhere.

The current flow is:

1. build or load a compact forecast request
2. convert that request into the expected feature order
3. run the forecast tool
4. convert the forecast into invoice pricing
5. send the invoice payload to Ollama/Qwen
6. write `.docx` plus metadata and audit logs

* * *

## Trade-offs

### 1. Upstream training repo vs thin serving repo

This repo is meant to sit on top of `Future-Sales-Forecast` rather than replace it.

#### Benefit:

- cleaner separation between training and serving
- easier MCP integration
- smaller serving codebase
- simpler invoice-focused iteration

#### Cost:

- model and feature assumptions can drift across repos
- artifact handoff becomes a maintenance concern
- serving quality depends on how cleanly upstream outputs are packaged

* * *

### 2. Small request payload vs richer real-world demand context

The current request shape is intentionally compact.

#### Benefit:

- easier tool calling
- faster inference
- simpler downstream invoice generation

#### Cost:

- limited store-level and company-level context
- weaker grounding for real procurement behaviour
- harder to generate consolidated, account-specific invoices

* * *

### 3. One invoice per item vs company-level grouped invoices

The current workflow produces one invoice per forecasted item request.

#### Benefit:

- easy to validate
- easy to audit
- straightforward artifact mapping between forecast and invoice

#### Cost:

- not how most real businesses issue invoices
- poor fit for buyer-level batching
- duplicated layout work when many items belong to the same company

* * *

### 4. Local Ollama generation vs faster operational latency

This repo currently leans on local Qwen generation for invoice text.

#### Benefit:

- flexible invoice language generation
- local control over the LLM path
- no hard dependency on a remote hosted provider

#### Cost:

- invoice latency is much higher than forecast latency
- operational performance depends heavily on local model speed
- invoice generation becomes the dominant bottleneck in end-to-end serving

* * *

## Future Improvements

The strongest next steps are:

- ~~keep the serving manifest and feature contract aligned with the trained `model.pkl` bundle from `Future-Sales-Forecast`~~
- move from single-item invoices to company-level grouped invoices
- attach real-world demand feeds instead of only sample or temporary payloads
- expose this as an actual long-running server that accepts pushed requests instead of mainly local command execution
- add request queueing, retries, and better MCP-facing observability around latency, failures, and artifact generation

* * *
