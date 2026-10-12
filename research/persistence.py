"""Durable SQLite persistence for ResearchPilot artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_runs_session_created
    ON research_runs(session_key, created_at);
CREATE INDEX IF NOT EXISTS idx_reports_session_created
    ON reports(session_key, created_at);
"""


class SQLiteStore:
    """Small transactional store for durable ResearchPilot data."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)

    def _connect(self) -> sqlite3.Connection:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA)
        connection.execute(
            """
            INSERT INTO schema_meta(key, value)
            VALUES ('schema_version', '2')
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """
        )
        return connection

    @staticmethod
    def _encode(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode(data: str) -> Any:
        return json.loads(data)

    def save_session(
        self,
        session_key: str,
        data: dict[str, Any],
        updated_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(session_key, data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    data=excluded.data,
                    updated_at=excluded.updated_at
                """,
                (session_key, self._encode(data), updated_at),
            )

    def load_session(self, session_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data FROM sessions WHERE session_key = ?",
                (session_key,),
            ).fetchone()

        if row is None:
            return None

        value = self._decode(row[0])
        return value if isinstance(value, dict) else None

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return all valid persisted sessions, newest first."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_key, data, updated_at FROM sessions "
                "ORDER BY updated_at DESC"
            ).fetchall()

        sessions = []
        for session_key, raw_data, updated_at in rows:
            try:
                value = self._decode(raw_data)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                value.setdefault("session_key", session_key)
                value.setdefault("updated_at", updated_at)
                sessions.append(value)
        return sessions

    def delete_session(self, session_key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE session_key = ?",
                (session_key,),
            )

    def save_artifact(
        self,
        table: str,
        session_key: str,
        data: Any,
        created_at: str,
    ) -> None:
        if table not in {"research_runs", "reports"}:
            raise ValueError("Unsupported persistence artifact table.")

        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO {table}(session_key, data, created_at) "
                "VALUES (?, ?, ?)",
                (session_key, self._encode(data), created_at),
            )

    def list_artifacts(
        self,
        table: str,
        session_key: str,
    ) -> list[Any]:
        if table not in {"research_runs", "reports"}:
            raise ValueError("Unsupported persistence artifact table.")

        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT data FROM {table} WHERE session_key = ? "
                "ORDER BY created_at, rowid",
                (session_key,),
            ).fetchall()

        return [self._decode(row[0]) for row in rows]
