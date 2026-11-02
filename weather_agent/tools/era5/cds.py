"""CDS catalogue discovery and resumable, source-native data acquisition.

No dataset, variable, level, date or grid allowlist lives here. Selections use
the dataset's own CDS API syntax, including MARS syntax for ERA5 complete.
"""
from __future__ import annotations

import calendar
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import urljoin
import zipfile

import requests
import xarray as xr

from ...cache import atomic_json


CATALOGUE_URL = "https://cds.climate.copernicus.eu/api/catalogue/v1/collections"
CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "foundation_live_cache" / "cds"


def _identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _offline(data_policy: str) -> bool:
    return data_policy == "cached" or os.getenv("WEATHER_AGENT_CACHE_ONLY", "").lower() in {
        "1", "true", "yes", "on",
    }


def _dataset_id(dataset: str) -> str:
    if not isinstance(dataset, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", dataset):
        raise ValueError("dataset must be a CDS dataset ID, e.g. reanalysis-era5-land")
    return dataset


def _metadata(url: str, data_policy: str, cache_root: Path) -> dict | list:
    path = cache_root / "catalogue" / (_identity(url) + ".json")
    if _offline(data_policy):
        try:
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError("cache_only_miss: CDS catalogue metadata are not cached") from error
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    value = response.json()
    atomic_json(path, {"url": url, "value": value})
    return value


def list_datasets(query: str = "era5", *, data_policy: str = "live") -> dict[str, Any]:
    """Discover the live CDS catalogue, following every pagination link."""
    url: str | None = CATALOGUE_URL
    seen: set[str] = set()
    collections = []
    while url and url not in seen:
        seen.add(url)
        page = _metadata(url, data_policy, CACHE_ROOT)
        for entry in page.get("collections", []):
            if query.casefold() in (entry.get("id", "") + " " + entry.get("title", "")).casefold():
                collections.append({key: entry.get(key) for key in (
                    "id", "title", "description", "extent", "license", "links",
                )})
        next_url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
        url = urljoin(url, next_url) if next_url else None
    return {"status": "ok", "datasets": collections, "source_url": CATALOGUE_URL}


def describe_dataset(dataset: str, *, data_policy: str = "live") -> dict[str, Any]:
    """Return source coverage and the complete download form/parameter choices."""
    url = CATALOGUE_URL + "/" + _dataset_id(dataset)
    collection = _metadata(url, data_policy, CACHE_ROOT)
    form_url = next((link["href"] for link in collection.get("links", []) if link.get("rel") == "form"), None)
    return {
        "status": "ok", "dataset": dataset, "collection": collection,
        "form": _metadata(urljoin(url, form_url), data_policy, CACHE_ROOT) if form_url else None,
        "source_url": url,
    }


def _values(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def split_selection(selection: dict[str, Any], split_by: str) -> list[dict[str, Any]]:
    """Partition actual calendar dates without creating a year/month/day cross product.

    Selections without a recognized date structure remain untouched. In
    particular, arbitrary MARS expressions can always use split_by='none'.
    """
    if split_by not in {"month", "day", "none"}:
        raise ValueError("split_by must be month, day or none")
    if split_by == "none":
        return [deepcopy(selection)]
    if "year" in selection and "month" in selection:
        result = []
        for year in _values(selection["year"]):
            for month in _values(selection["month"]):
                maximum = calendar.monthrange(int(year), int(month))[1]
                days = selection.get("day")
                valid = [str(day).zfill(2) for day in _values(days) if 1 <= int(day) <= maximum] if days is not None else None
                if days is not None and not valid:
                    continue
                groups = [[day] for day in valid] if split_by == "day" and valid else [valid]
                for group in groups:
                    item = deepcopy(selection)
                    item.update(year=[str(year)], month=[str(month).zfill(2)])
                    if group is not None:
                        item["day"] = group
                    result.append(item)
        if not result:
            raise ValueError("selection contains no valid calendar dates")
        return result
    expression = selection.get("date")
    if isinstance(expression, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}/to/\d{4}-\d{2}-\d{2}", expression):
        start_text, end_text = expression.split("/to/")
        cursor, end = date.fromisoformat(start_text), date.fromisoformat(end_text)
        if cursor > end:
            raise ValueError("date range starts after its end")
        result = []
        while cursor <= end:
            last = cursor if split_by == "day" else cursor.replace(day=calendar.monthrange(cursor.year, cursor.month)[1])
            last = min(last, end)
            result.append({**deepcopy(selection), "date": cursor.isoformat() if cursor == last else f"{cursor}/to/{last}"})
            cursor = last + timedelta(days=1)
        return result
    return [deepcopy(selection)]


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if not path.stat().st_size:
        raise RuntimeError("CDS returned an empty file")
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _client():
    # The shared loader also supports the deployed lightweight Python worker.
    from .backend import cds_client
    return cds_client()


def retrieve_data(
    dataset: str, selection: dict[str, Any], *, data_policy: str = "live",
    split_by: str = "month", cache_root: Path | None = None,
) -> dict[str, Any]:
    """Retrieve arbitrary source parameters, with atomic per-part cache publication."""
    dataset = _dataset_id(dataset)
    if not isinstance(selection, dict) or not selection:
        raise ValueError("selection must be a non-empty CDS API request mapping")
    # Copy/validate before I/O; cache identity includes every native parameter.
    selection = json.loads(json.dumps(selection, allow_nan=False))
    parts = split_selection(selection, split_by)
    root = Path(cache_root) if cache_root is not None else CACHE_ROOT
    files, records, hits = [], [], []
    client = None
    for part in parts:
        key = _identity({"version": 1, "dataset": dataset, "selection": part})
        directory = root / dataset / key
        target, manifest_path = directory / "download.bin", directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            hit = manifest.get("request_key") == key and manifest.get("fingerprint") == _fingerprint(target)
        except (OSError, ValueError, RuntimeError):
            hit = False
        if not hit:
            if _offline(data_policy):
                raise RuntimeError(f"cache_only_miss: CDS data absent for {dataset} request {key}")
            directory.mkdir(parents=True, exist_ok=True)
            if client is None:
                client = _client()
            fd, name = tempfile.mkstemp(prefix="download-", suffix=".part", dir=directory)
            os.close(fd)
            temporary = Path(name)
            try:
                client.retrieve(dataset, part, str(temporary))
                fingerprint = _fingerprint(temporary)
                os.replace(temporary, target)
                atomic_json(manifest_path, {
                    "request_key": key, "dataset": dataset, "selection": part,
                    "fingerprint": fingerprint,
                })
            finally:
                temporary.unlink(missing_ok=True)
        files.append(str(target))
        records.append({"request_key": key, "selection": part, "path": str(target), "cache_hit": hit})
        hits.append(hit)
    return {
        "status": "ok", "dataset": dataset, "selection": selection,
        "files": files, "parts": records, "cache_hit": all(hits),
        "source_url": CATALOGUE_URL + "/" + dataset,
    }


def data_paths(paths: list[str]) -> list[Path]:
    """Unpack CDS multi-file NetCDF/GRIB ZIP responses into content-keyed paths."""
    result = []
    for name in paths:
        path = Path(name)
        if not zipfile.is_zipfile(path):
            result.append(path)
            continue
        with zipfile.ZipFile(path) as archive:
            for position, entry in enumerate(archive.infolist()):
                if entry.is_dir():
                    continue
                # Use our own names; source member paths never reach the filesystem.
                suffix = Path(entry.filename).suffix.lower()
                if suffix not in {".nc", ".nc4", ".netcdf", ".grib", ".grb", ".grib2"}:
                    continue
                target = path.parent / f"{path.stem}-{position}-{entry.CRC:08x}{suffix}"
                if not target.is_file() or target.stat().st_size != entry.file_size:
                    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".part")
                    try:
                        with os.fdopen(fd, "wb") as destination, archive.open(entry) as source:
                            shutil.copyfileobj(source, destination)
                        os.replace(temporary, target)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
                result.append(target)
    if not result:
        raise RuntimeError("CDS response contains no NetCDF/GRIB members; use era5.retrieve for native files")
    return result


def open_fields(raw: dict[str, Any]) -> xr.Dataset:
    """Keep native variables, dimensions, units and all ensemble/model levels.

    Arrays remain lazy on disk; callers should select before materializing
    global or long-running requests. The Dataset owns its underlying handles.
    """
    sources = []
    try:
        for path in data_paths(raw["files"]):
            with path.open("rb") as handle:
                magic = handle.read(4)
            if magic == b"GRIB":
                try:
                    import cfgrib
                except ImportError as error:
                    raise RuntimeError("Reading GRIB requires cfgrib and ecCodes; era5.retrieve retains the native files") from error
                sources.extend(cfgrib.open_datasets(str(path), chunks="auto", backend_kwargs={"indexpath": ""}))
            else:
                # h5netcdf uses Python's Unicode-aware file handling on Windows.
                # netCDF4's C runtime cannot open this project's Chinese paths.
                source = xr.open_dataset(path, engine="scipy" if magic[:3] == b"CDF" else "h5netcdf")
                sources.append(source)
                # Keep the unchunked handle for reliable cleanup if optional
                # chunk-manager initialization fails.
                chunked = source.chunk("auto")
                chunked.set_close(source.close)
                sources[-1] = chunked
        dataset = sources[0].copy(deep=False) if len(sources) == 1 else xr.combine_by_coords(
            sources, combine_attrs="drop_conflicts", join="outer",
        )
        dataset.attrs.update(
            cds_dataset=raw["dataset"], source_url=raw["source_url"],
            cds_selection=json.dumps(raw["selection"], sort_keys=True), cache_hit=raw["cache_hit"],
        )
        dataset.set_close(lambda: [source.close() for source in sources])
        return dataset
    except Exception:
        for source in sources:
            source.close()
        raise
