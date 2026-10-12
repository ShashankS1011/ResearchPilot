# ResearchPilot

ResearchPilot is a Python command-line research assistant that searches the web with Tavily, extracts source content, cleans and ranks evidence, and synthesizes source-grounded answers with Google Gemini.

## Features

- Quick, deep, compare, and follow-up research modes
- Configurable source counts through environment variables
- Context-aware follow-up questions
- Tavily web search and extraction with retry/backoff
- Configurable retry jitter, retry budgets, and `Retry-After` handling
- Source-diversity filtering and duplicate-URL suppression
- Evidence cleaning, relevance ranking, confidence, and source-quality signals
- Duplicate evidence suppression and cross-source agreement signals
- Conservative cross-source contradiction detection and confidence penalties
- Gemini synthesis with deterministic evidence-grounded fallback
- Type-specific fallback protection against scraped list contamination
- Environment-based configuration
- Structured logging
- Persistent research/session JSON output
- Durable SQLite persistence for sessions, research runs, and reports
- Process-safe cache writes and resumable/cancellable workflow state
- JSON, Markdown, HTML, and CSV report exports
- Provider adapters with ordered failover support
- GitHub Actions tests across supported Python versions
- Optional authenticated HTTP API with health and metrics endpoints
- Automated pytest regression tests

## Setup

Use Python 3.11+.

```

## Durable persistence

ResearchPilot keeps the existing JSON exports for portability and also
stores sessions, research runs, and structured reports in SQLite. The
default database is `.researchpilot_data/researchpilot.db`.

Set `PERSISTENCE_ENABLED=false` to disable the durable store, or set
`PERSISTENCE_DB` to use a different database path. Existing JSON sessions
remain readable, and the SQLite store is used as a fallback when a JSON
session is unavailable.

## Reliability configuration

`REQUEST_TIMEOUT_SECONDS` controls provider request timeouts.
`RETRY_BASE_SECONDS` and `RETRY_MAX_SECONDS` control exponential backoff.
`RETRY_JITTER_SECONDS` adds optional random delay, while
`RETRY_BUDGET_SECONDS` limits total retry waiting time (`0` disables the
budget). Rate-limit responses with a valid `Retry-After` header are honored.

The command-line interface supports:

```powershell
python main.py --config
python main.py --history
python main.py --export report.html --format html
```

## Authenticated HTTP interface

Set a long random `WEB_AUTH_TOKEN` in `.env`, then start the server:

```powershell
python main.py --serve --host 127.0.0.1 --port 8080
```

`GET /health` is available for liveness checks. `GET /metrics` and
`POST /research` require:

```text
Authorization: Bearer <WEB_AUTH_TOKEN>
```

Example request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8080/research `
  -Method Post `
  -Headers @{Authorization="Bearer $env:WEB_AUTH_TOKEN"} `
  -ContentType "application/json" `
  -Body '{"query":"What is machine learning?","mode":"quick"}'
```

The built-in server binds to localhost by default. Put it behind HTTPS,
an access-controlled reverse proxy, and a stronger identity provider before
binding it to a public interface. Do not expose API keys through the web API.
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt