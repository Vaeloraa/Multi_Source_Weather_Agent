from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from ..agent.schema import normalize_tool_result


_GEO_CACHE_VERSION = "geolocation-cache-v2"
_GEO_CACHE_ROOT = Path(
    os.getenv("SEVIR_GEO_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sevir_world_geolocation_cache"))
)
_REQUEST_CACHE_VERSION = "request-tool-cache-v1"
_REQUEST_CACHE_MAX_BYTES = 2_000_000


def _request_cache_path(cache_key: str) -> Path | None:
    root = str(os.getenv("SEVIR_REQUEST_TOOL_CACHE_DIR", "")).strip()
    if not root:
        return None
    digest = hashlib.sha256(
        (_REQUEST_CACHE_VERSION + "\0" + cache_key).encode("utf-8")
    ).hexdigest()
    return Path(root) / f"{digest}.json"


def _read_request_cache(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return normalize_tool_result("request_cache", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def _write_request_cache(path: Path | None, result: Any) -> None:
    """Persist only genuinely JSON-safe, bounded results.

    ``default=str`` is deliberately not used: numpy arrays and model objects
    must remain in their native caches instead of becoming lossy strings.
    """
    if path is None:
        return
    encoded = _encode_request_cache(result)
    if encoded is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    except (OSError, TypeError, ValueError):
        return


def _encode_request_cache(result: Any) -> str | None:
    """Return a lossless bounded JSON payload, or ``None`` when unsafe."""
    try:
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return None
    if len(encoded.encode("utf-8")) > _REQUEST_CACHE_MAX_BYTES:
        return None
    return encoded


def resolve_event_id(value: Any) -> str:
    """Accept an event ID, one loaded dataset, or the injected ds list."""
    candidate = value
    if isinstance(candidate, (list, tuple)):
        if not candidate:
            raise ValueError("ds does not contain a SEVIR event")
        candidate = candidate[0]
    if isinstance(candidate, dict):
        candidate = candidate.get("event_id") or candidate.get("metadata", {}).get("event_id")
    event_id = str(candidate or "").strip()
    if not event_id:
        raise ValueError("a SEVIR event_id is required")
    return event_id


class WorldComponent:
    """Base class that records public API calls for reproducible agent traces."""

    def __init__(self, recorder: Callable[[str, dict[str, Any], Any], None]) -> None:
        self._recorder = recorder
        self._request_cache: dict[str, Any] = {}

    def _call(self, name: str, arguments: dict[str, Any], function: Callable[..., Any]) -> Any:
        cache_key = name + ":" + json.dumps(arguments, ensure_ascii=True, sort_keys=True, default=str)
        lookup = getattr(self._recorder, "lookup", None)
        replayed = lookup(name, arguments) if callable(lookup) else None
        if replayed is not None:
            self._request_cache[cache_key] = replayed
            try:
                self._recorder(name, arguments, replayed, cache_hit=True)
            except TypeError:
                self._recorder(name, arguments, replayed)
            return replayed
        if cache_key in self._request_cache:
            result = self._request_cache[cache_key]
            try:
                self._recorder(name, arguments, result, cache_hit=True)
            except TypeError:
                self._recorder(name, arguments, result)
            return result
        request_cache_path = _request_cache_path(cache_key)
        request_cached = _read_request_cache(request_cache_path)
        if request_cached is not None:
            self._request_cache[cache_key] = request_cached
            try:
                self._recorder(name, arguments, request_cached, cache_hit=True)
            except TypeError:
                self._recorder(name, arguments, request_cached)
            return request_cached
        # Geolocation is deterministic for a fixed event/projection/boundary
        # version, but every Agent repair runs in a fresh worker process.
        # Persist only geolocator results across workers; model outputs and
        # large arrays remain governed by their own caches.
        disk_path: Path | None = None
        if name.startswith("geolocator."):
            digest = hashlib.sha256(
                (_GEO_CACHE_VERSION + "\0" + cache_key).encode("utf-8")
            ).hexdigest()
            disk_path = _GEO_CACHE_ROOT / f"{digest}.json"
            try:
                if disk_path.is_file():
                    result = json.loads(disk_path.read_text(encoding="utf-8"))
                    self._request_cache[cache_key] = result
                    try:
                        self._recorder(name, arguments, result, cache_hit=True)
                    except TypeError:
                        self._recorder(name, arguments, result)
                    return result
            except (OSError, ValueError, TypeError):
                disk_path = None
        try:
            # Keep the established tool return objects unchanged.  Contract
            # descriptions belong in the prompt/trace, not in a global wrapper
            # that changes the value type seen by every existing caller.
            result = function(**arguments)
            # Keep the backend payload intact while adding machine-readable
            # aliases (for example ``forecast.frames`` and
            # ``ensemble.members``).  The returned object remains a dict, so
            # existing callers are not forced onto a new wrapper API.
            result = normalize_tool_result(name, result)
        except Exception as error:
            try:
                self._recorder(name, arguments, None, error)
            except TypeError:
                pass
            raise
        self._request_cache[cache_key] = result
        _write_request_cache(request_cache_path, result)
        if disk_path is not None:
            try:
                _GEO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
                temporary = disk_path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(
                    json.dumps(result, ensure_ascii=False, sort_keys=True, default=str),
                    encoding="utf-8",
                )
                temporary.replace(disk_path)
            except (OSError, TypeError, ValueError):
                pass
        backend_cache_hit = bool(result.get("cache_hit", False)) if isinstance(result, dict) else False
        try:
            self._recorder(name, arguments, result, cache_hit=backend_cache_hit)
        except TypeError:
            self._recorder(name, arguments, result)
        return result
