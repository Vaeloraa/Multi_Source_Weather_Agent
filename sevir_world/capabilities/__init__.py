"""Request-bound capabilities and their small semantic metadata values."""

from .models import (
    AdministrativeArea,
    Event,
    Place,
    UploadMatch,
)


def __getattr__(name: str):
    """Lazily expose capability classes without importing data backends."""

    if name in {"Aurora", "CasCast", "Era5", "Geospatial", "Knowledge", "Sevir"}:
        from .. import tools

        return getattr(tools, name)
    raise AttributeError(name)


__all__ = [
    "AdministrativeArea", "Aurora", "CasCast", "Era5", "Event", "Geospatial", "Knowledge",
    "Place", "Sevir",
    "UploadMatch",
]
