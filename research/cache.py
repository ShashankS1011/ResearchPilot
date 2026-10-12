"""Persistent JSON cache utilities for ResearchPilot."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from research.config import SETTINGS
from research.logging_config import logger


CacheClock = Callable[[], float]


class CacheStore:
    """Small persistent JSON cache with TTL support."""

    def __init__(
        self,
        directory: str | os.PathLike[str] | None = None,
        ttl_seconds: int | None = None,
        enabled: bool | None = None,
        clock: CacheClock | None = None,
    ) -> None:
        """Initialize a cache store."""

        self.directory = Path(
            directory
            if directory is not None
            else SETTINGS.cache_dir
        )

        self.ttl_seconds = (
            SETTINGS.cache_ttl_seconds
            if ttl_seconds is None
            else max(
                int(ttl_seconds),
                0,
            )
        )

        self.enabled = (
            SETTINGS.cache_enabled
            if enabled is None
            else bool(enabled)
        )

        self._clock = (
            clock
            if clock is not None
            else time.time
        )

    @staticmethod
    def build_key(
        namespace: str,
        value: Any,
    ) -> str:
        """
        Build a deterministic cache key.

        The serialized value is hashed so arbitrary query/input
        data does not become part of the cache filename.
        """

        normalized_namespace = str(
            namespace or "default"
        ).strip().lower()

        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            serialized = repr(value)

        payload = (
            f"{normalized_namespace}:{serialized}"
        ).encode(
            "utf-8"
        )

        digest = hashlib.sha256(
            payload
        ).hexdigest()

        return (
            f"{normalized_namespace}-{digest}"
        )

    def _path_for_key(
        self,
        key: str,
    ) -> Path:
        """Return the filesystem path for a cache key."""

        safe_key = str(
            key or ""
        ).strip()

        if not safe_key:
            raise ValueError(
                "Cache key cannot be empty."
            )

        return self.directory / f"{safe_key}.json"

    @contextmanager
    def _lock(self, path: Path):
        """Serialize access to one cache key across processes."""

        lock_path = path.with_suffix(".lock")
        lock_file = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            lock_file = lock_path.open("a+", encoding="utf-8")
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if lock_file is not None:
                try:
                    if os.name == "nt":
                        import msvcrt

                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_file.close()

    def _is_expired(
        self,
        created_at: float,
    ) -> bool:
        """Return whether a cache entry has expired."""

        if self.ttl_seconds <= 0:
            return False

        age = self._clock() - created_at

        return age >= self.ttl_seconds

    def get(
        self,
        key: str,
    ) -> Any | None:
        """
        Return a cached value.

        A missing, expired, malformed, or corrupted entry is treated
        as a cache miss.
        """

        if not self.enabled:
            logger.debug(
                "Cache disabled; skipping lookup."
            )
            return None

        path = self._path_for_key(
            key
        )

        try:
            with self._lock(path):
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    entry = json.load(file)

        except FileNotFoundError:
            logger.debug(
                "Cache miss: %s",
                key,
            )
            return None

        except (
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as error:
            logger.warning(
                "Ignoring corrupted cache entry %s: %s",
                path,
                error,
            )

            with self._lock(path):
                self._remove_entry(path)

            return None

        if not isinstance(
            entry,
            dict,
        ):
            logger.warning(
                "Ignoring malformed cache entry: %s",
                path,
            )

            with self._lock(path):
                self._remove_entry(path)

            return None

        created_at = entry.get(
            "created_at"
        )

        if not isinstance(
            created_at,
            (int, float),
        ):
            logger.warning(
                "Ignoring cache entry with invalid timestamp: %s",
                path,
            )

            with self._lock(path):
                self._remove_entry(path)

            return None

        if self._is_expired(
            float(created_at)
        ):
            logger.debug(
                "Cache entry expired: %s",
                key,
            )

            with self._lock(path):
                self._remove_entry(path)

            return None

        if "value" not in entry:
            logger.warning(
                "Ignoring cache entry without value: %s",
                path,
            )

            with self._lock(path):
                self._remove_entry(path)

            return None

        logger.debug(
            "Cache hit: %s",
            key,
        )

        return entry["value"]

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Persist a value in the cache atomically."""

        if not self.enabled:
            logger.debug(
                "Cache disabled; skipping write."
            )
            return

        path = self._path_for_key(
            key
        )

        try:
            self.directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            entry = {
                "created_at": self._clock(),
                "value": value,
            }

            with self._lock(path):
                self._atomic_write(path, entry)

            logger.debug(
                "Cache write: %s",
                key,
            )

        except (
            OSError,
            TypeError,
            ValueError,
        ) as error:
            # Cache failures should never make an otherwise successful
            # research operation fail.
            logger.warning(
                "Could not write cache entry %s: %s",
                key,
                error,
            )

    def delete(
        self,
        key: str,
    ) -> None:
        """Delete a cache entry if it exists."""

        path = self._path_for_key(
            key
        )

        with self._lock(path):
            self._remove_entry(path)

    def clear(self) -> None:
        """Delete all JSON cache entries."""

        if not self.directory.exists():
            return

        try:
            for path in self.directory.glob(
                "*.json"
            ):
                self._remove_entry(
                    path
                )

        except OSError as error:
            logger.warning(
                "Could not clear cache directory %s: %s",
                self.directory,
                error,
            )

    @staticmethod
    def _remove_entry(
        path: Path,
    ) -> None:
        """Remove a cache entry without propagating filesystem errors."""

        try:
            path.unlink(
                missing_ok=True
            )
        except OSError as error:
            logger.warning(
                "Could not remove cache entry %s: %s",
                path,
                error,
            )

    @staticmethod
    def _atomic_write(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        """
        Atomically replace a cache file.

        A temporary file in the same directory is used so that a
        process interruption cannot leave a partially written JSON
        document at the final cache path.
        """

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(
                    file.name
                )

                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )

                file.flush()
                os.fsync(
                    file.fileno()
                )

            os.replace(
                temporary_path,
                path,
            )

            temporary_path = None

        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass


def get_cache(
    namespace: str,
    value: Any,
) -> tuple[CacheStore, str]:
    """
    Return the default cache store and deterministic key.

    This helper is intended for external-service integrations.
    """

    store = CacheStore()

    key = store.build_key(
        namespace,
        value,
    )

    return store, key