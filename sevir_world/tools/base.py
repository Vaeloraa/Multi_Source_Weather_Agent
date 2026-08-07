from __future__ import annotations

from typing import Any, Callable


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

    def _call(self, name: str, arguments: dict[str, Any], function: Callable[..., Any]) -> Any:
        try:
            result = function(**arguments)
        except Exception as error:
            try:
                self._recorder(name, arguments, None, error)
            except TypeError:
                pass
            raise
        self._recorder(name, arguments, result)
        return result
