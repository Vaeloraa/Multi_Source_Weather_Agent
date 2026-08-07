from __future__ import annotations

from typing import Any, Callable


class WorldComponent:
    """Base class that records public API calls for reproducible agent traces."""

    def __init__(self, recorder: Callable[[str, dict[str, Any], Any], None]) -> None:
        self._recorder = recorder

    def _call(self, name: str, arguments: dict[str, Any], function: Callable[..., Any]) -> Any:
        result = function(**arguments)
        self._recorder(name, arguments, result)
        return result
