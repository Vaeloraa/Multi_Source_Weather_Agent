"""Lazy request-bound tools exposed to generated Python programs."""

from importlib import import_module

_TOOLS = {
    "Aurora": (".aurora.Aurora", "Aurora"),
    "CasCast": (".cascast.CasCast", "CasCast"),
    "Geolocator": (".geolocator.Geolocator", "Geolocator"),
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
