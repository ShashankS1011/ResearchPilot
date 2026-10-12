"""Minimal authenticated HTTP interface for local or private deployments."""

from __future__ import annotations

import json
import os
from time import monotonic
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from main import perform_research, generate_and_display_answer
from research.metrics import METRICS


def _authorized(handler: BaseHTTPRequestHandler) -> bool:
    token = os.getenv("WEB_AUTH_TOKEN", "").strip()
    if not token:
        return False
    return handler.headers.get("Authorization", "") == f"Bearer {token}"


def create_handler(
    research_fn: Callable[[str, str], Any] | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Create a request handler with an injectable research function."""

    execute = research_fn or _run_research

    class ResearchPilotHandler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok"})
                return
            if self.path == "/metrics":
                if not _authorized(self):
                    self._json(401, {"error": "authentication required"})
                    return
                self._json(200, METRICS.snapshot())
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/research":
                self._json(404, {"error": "not found"})
                return
            if not _authorized(self):
                self._json(401, {"error": "authentication required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 64 * 1024:
                    raise ValueError("request body is too large")
                payload = json.loads(self.rfile.read(length))
                query = str(payload.get("query", "")).strip()
                mode = str(payload.get("mode", "quick")).strip().lower()
                if not query or len(query) > 2000:
                    raise ValueError("query must contain 1-2000 characters")
                if mode not in {"quick", "deep"}:
                    raise ValueError("mode must be quick or deep")
                started_at = monotonic()
                try:
                    report = execute(query, mode)
                except Exception:
                    METRICS.observe("web.research", started_at, False)
                    raise
                METRICS.observe("web.research", started_at, True)
                self._json(200, {"report": report})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
            except Exception:
                self._json(500, {"error": "research request failed"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ResearchPilotHandler


def _run_research(query: str, mode: str) -> dict[str, Any]:
    sources = perform_research(query, mode=mode)
    if not sources:
        return {"executive_answer": "", "key_findings": [], "limitations": []}
    generate_and_display_answer(query, sources, mode=mode)
    from research.storage import load_report_history

    reports = load_report_history()
    return reports[-1] if reports else {}


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the authenticated ResearchPilot HTTP server."""

    if not os.getenv("WEB_AUTH_TOKEN", "").strip():
        raise RuntimeError("WEB_AUTH_TOKEN must be configured before serving.")
    server = ThreadingHTTPServer((host, port), create_handler())
    server.serve_forever()
