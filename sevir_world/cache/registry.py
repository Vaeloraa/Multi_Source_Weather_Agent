"""Process-safe metadata registry for reusable scientific cache artifacts.

The registry deliberately stores metadata and leases, not large weather
arrays.  Artifacts may remain on local disk today and move to an object store
later without changing the lookup contract used by callers.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Iterator, Mapping

from ..config import ROOT


REGISTRY_SCHEMA_VERSION = "cache-registry-v1"
RESOLVE_TELEMETRY_SCHEMA_VERSION = "cache-resolve-telemetry-v1"
RESOLVE_OUTCOMES = ("hit", "miss", "stale", "invalid", "superseded")
DEFAULT_REGISTRY_RELATIVE_PATH = Path("data") / "runtime" / "cache_registry.sqlite3"
_SCHEMA_LOCK = threading.RLock()


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(value[key]) for key in sorted(value)}
    if isinstance(value, set):
        cleaned = [_clean(item) for item in value]
        return sorted(
            cleaned,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def versioned_artifact_identity(
    namespace: str,
    *,
    source: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    pipeline_version: str = "",
    model: str = "",
    model_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a content-sensitive identity for one scientific artifact.

    A caller can begin with cheap file-stat fingerprints and later replace
    them with checksums.  Either way, model and preprocessing changes become
    part of the key instead of silently reusing an incompatible file.
    """

    canonical = _clean({
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "namespace": str(namespace).strip(),
        "source": source or {},
        "parameters": parameters or {},
        "pipeline_version": str(pipeline_version or ""),
        "model": str(model or ""),
        "model_fingerprint": model_fingerprint or {},
    })
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "request_key": hashlib.sha256(encoded).hexdigest(),
        "canonical": canonical,
    }


def registry_entry_key(namespace: str, identity: Mapping[str, Any]) -> str:
    """Build the stable primary key used by the SQLite registry."""

    request_key = str(identity.get("request_key") or "").strip()
    if not request_key:
        raise ValueError("cache identity request_key is required")
    encoded = f"{str(namespace).strip()}\0{request_key}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def registry_path(root: str | Path | None = None) -> Path:
    configured = os.getenv("SEVIR_CACHE_REGISTRY_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(root or ROOT) / DEFAULT_REGISTRY_RELATIVE_PATH


def file_stat_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return a cheap invalidation fingerprint without reading a large file."""

    item = Path(path)
    try:
        stat = item.stat()
    except OSError:
        return {"path": str(item), "exists": False}
    return {
        "path": str(item.resolve()),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class CacheRegistry:
    """SQLite-backed index shared by processes on one host.

    SQLite WAL gives concurrent readers and short atomic writes.  It is not a
    multi-host distributed cache; the backend field and artifact URI keep that
    later migration explicit.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else registry_path()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        with _SCHEMA_LOCK:
            for attempt in range(20):
                try:
                    mode = str(
                        connection.execute("PRAGMA journal_mode").fetchone()[0]
                    ).lower()
                    if mode != "wal":
                        connection.execute("PRAGMA journal_mode = WAL")
                    connection.execute("PRAGMA synchronous = NORMAL")
                    self._initialize(connection)
                    break
                except sqlite3.OperationalError as error:
                    if "locked" not in str(error).lower() or attempt == 19:
                        connection.close()
                        raise
                    time.sleep(0.05 * (attempt + 1))
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                entry_key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                request_key TEXT NOT NULL,
                artifact_uri TEXT NOT NULL,
                backend TEXT NOT NULL,
                status TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                expires_at REAL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                identity_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cache_namespace_status
                ON cache_entries(namespace, status);
            CREATE INDEX IF NOT EXISTS idx_cache_artifact_uri
                ON cache_entries(artifact_uri);
            CREATE INDEX IF NOT EXISTS idx_cache_last_access
                ON cache_entries(last_accessed_at);
            CREATE TABLE IF NOT EXISTS cache_leases (
                entry_key TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                acquired_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cache_resolve_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                entry_key TEXT NOT NULL,
                outcome TEXT NOT NULL,
                observed_at REAL NOT NULL,
                latency_ms REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cache_resolve_namespace_time
                ON cache_resolve_events(namespace, observed_at);
            CREATE INDEX IF NOT EXISTS idx_cache_resolve_outcome_time
                ON cache_resolve_events(outcome, observed_at);
            """
        )

    @staticmethod
    def _record_resolve_event(
        connection: sqlite3.Connection,
        *,
        namespace: str,
        entry_key: str,
        outcome: str,
        observed_at: float,
        started_monotonic: float,
    ) -> None:
        if outcome not in RESOLVE_OUTCOMES:
            raise ValueError(f"unsupported cache resolve outcome: {outcome!r}")
        latency_ms = max(0.0, (time.monotonic() - started_monotonic) * 1000.0)
        connection.execute(
            """
            INSERT INTO cache_resolve_events(
                namespace, entry_key, outcome, observed_at, latency_ms
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(namespace).strip(),
                str(entry_key),
                str(outcome),
                float(observed_at),
                float(latency_ms),
            ),
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for field in ("identity_json", "metadata_json"):
            try:
                result[field.removesuffix("_json")] = json.loads(result.pop(field))
            except (TypeError, ValueError):
                result[field.removesuffix("_json")] = {}
                result.pop(field, None)
        return result

    def register(
        self,
        namespace: str,
        identity: Mapping[str, Any],
        artifact: str | Path,
        *,
        backend: str = "filesystem",
        ttl_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        namespace = str(namespace).strip()
        if not namespace:
            raise ValueError("cache namespace is required")
        request_key = str(identity.get("request_key") or "").strip()
        if not request_key:
            raise ValueError("cache identity request_key is required")
        timestamp = float(time.time() if now is None else now)
        item = Path(artifact)
        artifact_uri = str(item.resolve()) if backend == "filesystem" else str(artifact)
        try:
            size_bytes = int(item.stat().st_size) if backend == "filesystem" else 0
        except OSError:
            size_bytes = 0
        entry_key = registry_entry_key(namespace, identity)
        expires_at = timestamp + float(ttl_seconds) if ttl_seconds is not None else None
        identity_json = json.dumps(_clean(identity), ensure_ascii=False, sort_keys=True)
        metadata_json = json.dumps(_clean(metadata or {}), ensure_ascii=False, sort_keys=True)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # The same local filename cannot safely represent two model/input
            # identities after an overwrite. Keep the history but make the old
            # record ineligible for reads.
            connection.execute(
                """
                UPDATE cache_entries
                SET status = 'superseded', updated_at = ?
                WHERE namespace = ? AND artifact_uri = ? AND entry_key <> ?
                  AND status = 'available'
                """,
                (timestamp, namespace, artifact_uri, entry_key),
            )
            connection.execute(
                """
                INSERT INTO cache_entries (
                    entry_key, namespace, request_key, artifact_uri, backend,
                    status, size_bytes, created_at, updated_at,
                    last_accessed_at, expires_at, hit_count,
                    identity_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, 'available', ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(entry_key) DO UPDATE SET
                    artifact_uri = excluded.artifact_uri,
                    backend = excluded.backend,
                    status = 'available',
                    size_bytes = excluded.size_bytes,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at,
                    identity_json = excluded.identity_json,
                    metadata_json = excluded.metadata_json
                """,
                (
                    entry_key, namespace, request_key, artifact_uri, backend,
                    size_bytes, timestamp, timestamp, timestamp, expires_at,
                    identity_json, metadata_json,
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM cache_entries WHERE entry_key = ?", (entry_key,)
            ).fetchone()
        if row is None:  # pragma: no cover - defensive SQLite invariant
            raise RuntimeError("cache registry write did not persist")
        return self._row(row)

    def resolve(
        self,
        namespace: str,
        identity: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        started_monotonic = time.monotonic()
        timestamp = float(time.time() if now is None else now)
        normalized_namespace = str(namespace).strip()
        entry_key = registry_entry_key(normalized_namespace, identity)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM cache_entries WHERE entry_key = ? AND namespace = ?",
                (entry_key, normalized_namespace),
            ).fetchone()
            if row is None:
                self._record_resolve_event(
                    connection,
                    namespace=normalized_namespace,
                    entry_key=entry_key,
                    outcome="miss",
                    observed_at=timestamp,
                    started_monotonic=started_monotonic,
                )
                connection.commit()
                return None
            if row["status"] != "available":
                stored_status = str(row["status"])
                outcome = {
                    "expired": "stale",
                    "missing": "invalid",
                    "superseded": "superseded",
                }.get(stored_status, "invalid")
                self._record_resolve_event(
                    connection,
                    namespace=normalized_namespace,
                    entry_key=entry_key,
                    outcome=outcome,
                    observed_at=timestamp,
                    started_monotonic=started_monotonic,
                )
                connection.commit()
                return None
            status = "available"
            if row["expires_at"] is not None and float(row["expires_at"]) <= timestamp:
                status = "expired"
            elif row["backend"] == "filesystem" and not Path(row["artifact_uri"]).exists():
                status = "missing"
            if status != "available":
                connection.execute(
                    "UPDATE cache_entries SET status = ?, updated_at = ? WHERE entry_key = ?",
                    (status, timestamp, entry_key),
                )
                self._record_resolve_event(
                    connection,
                    namespace=normalized_namespace,
                    entry_key=entry_key,
                    outcome="stale" if status == "expired" else "invalid",
                    observed_at=timestamp,
                    started_monotonic=started_monotonic,
                )
                connection.commit()
                return None
            connection.execute(
                """
                UPDATE cache_entries
                SET last_accessed_at = ?, hit_count = hit_count + 1
                WHERE entry_key = ?
                """,
                (timestamp, entry_key),
            )
            updated = connection.execute(
                "SELECT * FROM cache_entries WHERE entry_key = ?", (entry_key,)
            ).fetchone()
            self._record_resolve_event(
                connection,
                namespace=normalized_namespace,
                entry_key=entry_key,
                outcome="hit",
                observed_at=timestamp,
                started_monotonic=started_monotonic,
            )
            # Keep the final read inside the same IMMEDIATE transaction. A
            # concurrent register cannot supersede this row between commit
            # and the value returned to the caller.
            connection.commit()
        return self._row(updated) if updated is not None else None

    def resolve_telemetry(
        self,
        *,
        namespace: str | None = None,
        since: float | None = None,
    ) -> dict[str, Any]:
        """Aggregate real cache lookups, including misses and invalidations.

        ``inventory().hit_count`` measures successful entry reuse over each
        entry's lifetime.  This method instead measures the request-side
        denominator, so hit rate and lookup latency are no longer inferred
        from files that happen to exist on disk.
        """

        clauses: list[str] = []
        values: list[Any] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            values.append(str(namespace).strip())
        if since is not None:
            clauses.append("observed_at >= ?")
            values.append(float(since))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT namespace, outcome, observed_at, latency_ms
                FROM cache_resolve_events
                """ + where + " ORDER BY event_id",
                values,
            ).fetchall()

        counts = {outcome: 0 for outcome in RESOLVE_OUTCOMES}
        namespace_counts: dict[str, dict[str, int]] = {}
        latencies: list[float] = []
        first_observed_at: float | None = None
        last_observed_at: float | None = None
        for row in rows:
            outcome = str(row["outcome"])
            if outcome not in counts:
                continue
            item_namespace = str(row["namespace"])
            counts[outcome] += 1
            group = namespace_counts.setdefault(
                item_namespace, {name: 0 for name in RESOLVE_OUTCOMES}
            )
            group[outcome] += 1
            latencies.append(max(0.0, float(row["latency_ms"])))
            observed_at = float(row["observed_at"])
            first_observed_at = (
                observed_at
                if first_observed_at is None
                else min(first_observed_at, observed_at)
            )
            last_observed_at = (
                observed_at
                if last_observed_at is None
                else max(last_observed_at, observed_at)
            )

        def percentile(values_to_measure: list[float], quantile: float) -> float | None:
            if not values_to_measure:
                return None
            ordered = sorted(values_to_measure)
            index = int(round((len(ordered) - 1) * quantile))
            return float(ordered[index])

        total = sum(counts.values())
        namespaces: dict[str, Any] = {}
        for item_namespace, group_counts in sorted(namespace_counts.items()):
            group_total = sum(group_counts.values())
            namespaces[item_namespace] = {
                "lookups": group_total,
                "outcomes": group_counts,
                "hit_rate": (
                    float(group_counts["hit"]) / float(group_total)
                    if group_total else None
                ),
            }
        return {
            "schema_version": RESOLVE_TELEMETRY_SCHEMA_VERSION,
            "scope": {
                "namespace": str(namespace).strip() if namespace is not None else None,
                "since": float(since) if since is not None else None,
            },
            "lookups": total,
            "outcomes": counts,
            "hit_rate": float(counts["hit"]) / float(total) if total else None,
            "latency_ms": {
                "mean": (sum(latencies) / len(latencies)) if latencies else None,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "max": max(latencies) if latencies else None,
            },
            "first_observed_at": first_observed_at,
            "last_observed_at": last_observed_at,
            "namespaces": namespaces,
        }

    def inventory(self) -> dict[str, Any]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT namespace, status, COUNT(*) AS entry_count,
                       COALESCE(SUM(size_bytes), 0) AS size_bytes,
                       COALESCE(SUM(hit_count), 0) AS hit_count
                FROM cache_entries
                GROUP BY namespace, status
                ORDER BY namespace, status
                """
            ).fetchall()
            physical_rows = connection.execute(
                """
                SELECT artifact_uri, backend, size_bytes, status
                FROM cache_entries
                """
            ).fetchall()
        groups = [dict(row) for row in rows]
        physical_sizes: dict[str, int] = {}
        available_sizes: dict[str, int] = {}
        for row in physical_rows:
            uri = str(row["artifact_uri"])
            if str(row["backend"]) == "filesystem":
                try:
                    size = int(Path(uri).stat().st_size)
                except OSError:
                    size = 0
            else:
                size = max(0, int(row["size_bytes"]))
            physical_sizes[uri] = max(physical_sizes.get(uri, 0), size)
            if str(row["status"]) == "available":
                available_sizes[uri] = max(available_sizes.get(uri, 0), size)
        logical_size = sum(
            int(row["size_bytes"]) for row in groups if row["status"] == "available"
        )
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "registry_path": str(self.path.resolve()),
            "groups": groups,
            "entry_count": sum(int(row["entry_count"]) for row in groups),
            # Physical bytes include expired/superseded registry rows while
            # their files still occupy storage. ``available_size_bytes`` is
            # retained separately for cache-serving capacity.
            "size_bytes": sum(physical_sizes.values()),
            "available_size_bytes": sum(available_sizes.values()),
            "logical_size_bytes": logical_size,
            "hit_count": sum(int(row["hit_count"]) for row in groups),
        }

    def acquire_lease(
        self,
        entry_key: str,
        owner: str,
        *,
        lease_seconds: float,
        now: float | None = None,
    ) -> bool:
        """Acquire an expiring cross-process singleflight lease."""

        timestamp = float(time.time() if now is None else now)
        expires_at = timestamp + max(1.0, float(lease_seconds))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM cache_leases WHERE expires_at <= ?", (timestamp,))
            row = connection.execute(
                "SELECT owner FROM cache_leases WHERE entry_key = ?", (str(entry_key),)
            ).fetchone()
            if row is not None and str(row["owner"]) != str(owner):
                connection.commit()
                return False
            connection.execute(
                """
                INSERT INTO cache_leases(entry_key, owner, acquired_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entry_key) DO UPDATE SET
                    owner = excluded.owner,
                    acquired_at = excluded.acquired_at,
                    expires_at = excluded.expires_at
                """,
                (str(entry_key), str(owner), timestamp, expires_at),
            )
            connection.commit()
        return True

    def release_lease(self, entry_key: str, owner: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM cache_leases WHERE entry_key = ? AND owner = ?",
                (str(entry_key), str(owner)),
            )
            connection.commit()
        return bool(cursor.rowcount)
