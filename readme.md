# ResearchPilot

[![ResearchPilot CI](https://github.com/ShashankS1011/ResearchPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/ShashankS1011/ResearchPilot/actions/workflows/ci.yml)

ResearchPilot is a source-grounded research assistant for the command line.
It searches the web with Tavily, extracts and evaluates evidence, and
synthesizes concise answers with Google Gemini. The application is designed
to remain useful when providers fail: it validates responses, retries
transient errors, persists research artifacts, and falls back to
evidence-based reports when synthesis is unavailable.

## Highlights

- Quick, deep, compare, and follow-up research workflows
- Query planning and dependency-aware workflow execution
- Tavily search and extraction with timeout enforcement
- Evidence cleaning, relevance ranking, source-quality scoring, and deduplication
- Cross-source agreement and conservative contradiction detection
- Citation-aware Gemini synthesis with deterministic fallback reports
- Configurable retries, exponential backoff, jitter, retry budgets, and `Retry-After`
- Persistent JSON artifacts and SQLite session/report history
- Process-safe, TTL-aware cache with atomic writes
- JSON, Markdown, HTML, and CSV report exports
- Optional authenticated HTTP API with health and metrics endpoints
- Provider protocols, adapters, and ordered failover support
- GitHub Actions CI for Python 3.11, 3.12, and 3.13

## Requirements

- Python 3.11 or newer
- A [Tavily API key](https://tavily.com/)
- A [Google Gemini API key](https://ai.google.dev/)

## Installation

PowerShell:

```powershell
git clone https://github.com/ShashankS1011/ResearchPilot.git
Set-Location ResearchPilot

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell execution policy prevents activation, use the virtual
environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

## Configuration

Create a local `.env` file from [`.env.example`](.env.example):

```powershell
Copy-Item .env.example .env
notepad .env
```

Set the required provider credentials:

```env
TAVILY_API_KEY=your_tavily_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Never commit `.env`. It is excluded by [`.gitignore`](.gitignore).

Important runtime settings include:

| Setting | Purpose | Default |
| --- | --- | --- |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.8-flash` |
| `QUICK_SOURCES` | Sources for quick research | `3` |
| `DEEP_SOURCES` | Sources for deep research | `8` |
| `REQUEST_TIMEOUT_SECONDS` | Provider request timeout | `45` |
| `TAVILY_MAX_RETRIES` | Tavily retry count | `2` |
| `GEMINI_MAX_RETRIES` | Gemini retry count | `2` |
| `RETRY_BASE_SECONDS` | Initial retry delay | `1.5` |
| `RETRY_MAX_SECONDS` | Maximum retry delay | `30.0` |
| `RETRY_JITTER_SECONDS` | Optional random retry delay | `0` |
| `RETRY_BUDGET_SECONDS` | Total retry waiting-time limit; `0` disables it | `0` |
| `CACHE_TTL_SECONDS` | Cache lifetime; `0` means no expiration | `86400` |
| `PERSISTENCE_ENABLED` | Enable SQLite persistence | `true` |
| `PERSISTENCE_DB` | SQLite database path | `.researchpilot_data/researchpilot.db` |
| `WEB_AUTH_TOKEN` | Token required by the HTTP API | unset |

Run `python main.py --config` to display safe, non-secret runtime settings.

## Command-line usage

Run an interactive session:

```powershell
python main.py
```

Run a question directly:

```powershell
python main.py "What is machine learning?"
python main.py --deep "What are the latest advances in battery technology?"
```

Inspect configuration and saved history:

```powershell
python main.py --config
python main.py --history
```

Export the latest structured report:

```powershell
python main.py --export report.json --format json
python main.py --export report.md --format markdown
python main.py --export report.html --format html
python main.py --export report.csv --format csv
```

Reports and sessions are written to the project directory by default.
Generated files and local databases are ignored by Git.

## HTTP API

The optional built-in HTTP server is intended for local or private
deployments. Configure a strong random token first:

```powershell
$env:WEB_AUTH_TOKEN = "replace-with-a-long-random-token"
python main.py --serve --host 127.0.0.1 --port 8080
```

Endpoints:

| Method | Path | Authentication | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | None | Liveness check |
| `GET` | `/metrics` | Bearer token | Request counts, failures, and latency |
| `POST` | `/research` | Bearer token | Run quick or deep research |

Example request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8080/research `
  -Method Post `
  -Headers @{ Authorization = "Bearer $env:WEB_AUTH_TOKEN" } `
  -ContentType "application/json" `
  -Body '{"query":"What is machine learning?","mode":"quick"}'
```

The API validates request size, query length, and research mode. The default
bind address is localhost. Before exposing the service publicly, place it
behind HTTPS, a reverse proxy, rate limiting, network controls, and a
production identity provider. Do not expose provider API keys to clients.

## Persistence and caching

ResearchPilot uses two local persistence mechanisms:

- JSON files provide portable research, report, and session artifacts.
- SQLite stores durable sessions, research runs, reports, indexes, and schema
  metadata.
- The file cache stores successful provider responses with TTL validation.
- Cache writes are atomic and protected against concurrent process access.

Disable SQLite persistence with:

```env
PERSISTENCE_ENABLED=false
```

Use a different database with:

```env
PERSISTENCE_DB=C:\path\to\researchpilot.db
```

## Reliability behavior

Provider requests use bounded retries for transient failures such as
timeouts, connection errors, rate limits, and server errors. A valid
`Retry-After` response is honored. Failed or malformed responses are not
cached. Cache and persistence write failures are surfaced through structured
logs without turning a successful provider response into a failed research
run.

If Gemini is unavailable or quota-limited, ResearchPilot creates an
evidence-based fallback report instead of returning an empty or fabricated
answer.

## Development and testing

Run the complete local test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Compile the source and tests:

```powershell
.\.venv\Scripts\python.exe -m compileall -q research tests
```

The project configuration is in [`pyproject.toml`](pyproject.toml). GitHub
Actions runs the tests and compilation checks on every push and pull request,
and runs `pip-audit` to check dependencies for known vulnerabilities.

## Project structure

```text
ResearchPilot/
├── main.py                 # CLI entry point and research orchestration
├── research/
│   ├── search.py           # Tavily search and retry handling
│   ├── extract.py          # Tavily extraction and validation
│   ├── llm.py              # Gemini synthesis and fallback integration
│   ├── processor.py        # Evidence cleaning and intelligence
│   ├── planner.py          # Research plan generation
│   ├── workflow.py         # Resumable workflow state and execution
│   ├── persistence.py      # SQLite storage
│   ├── storage.py          # JSON/SQLite artifact storage and exports
│   ├── web.py              # Authenticated HTTP interface
│   └── ui.py               # Terminal presentation
├── tests/                  # Regression and reliability tests
├── .env.example            # Configuration template
└── .github/workflows/ci.yml
```

## Security notes

- Keep API keys and `WEB_AUTH_TOKEN` in local environment configuration.
- Do not commit `.env`, databases, caches, reports, or virtual environments.
- Use a strong, unique HTTP token for private API deployments.
- Use HTTPS and a production authentication provider for public deployments.
- Review Tavily and Google AI dashboards for quota, usage, and billing limits.

## License

No license has been selected for this repository yet. Add an explicit
license before distributing ResearchPilot or accepting external
contributions.
