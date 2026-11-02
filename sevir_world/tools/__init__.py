"""Lazy request-bound tools exposed to generated Python programs."""

from importlib import import_module

_TOOLS = {
    "Knowledge": (".knowledge.Knowledge", "Knowledge"),
    "Aurora": (".aurora.Aurora", "Aurora"),
    "Era5": (".era5.Era5", "Era5"),
    "CasCast": (".cascast.CasCast", "CasCast"),
    "Geospatial": (".geospatial.tool", "Geospatial"),
    "Sevir": (".sevir.Sevir", "Sevir"),
}


def __getattr__(name: str):
    """Load one capability module on first package attribute access.

    Parameters
    ----------
    name : str
        Capability class name.
    """

    if name not in _TOOLS:
        raise AttributeError(name)
    module_name, class_name = _TOOLS[name]
    value = getattr(import_module(module_name, __name__), class_name)
    globals()[name] = value
    return value


__all__ = list(_TOOLS)
