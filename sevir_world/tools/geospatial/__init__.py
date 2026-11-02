from .contracts import BoundingBox, GeoPoint, GridPoint, Polygon, Region


def __getattr__(name: str):
    if name == "Geospatial":
        from .tool import Geospatial
        return Geospatial
    raise AttributeError(name)

__all__ = ["BoundingBox", "GeoPoint", "Geospatial", "GridPoint", "Polygon", "Region"]
