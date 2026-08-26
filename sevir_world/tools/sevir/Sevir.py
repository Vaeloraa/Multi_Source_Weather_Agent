from __future__ import annotations

"""Public SEVIR tool and its request-bound catalog/observation interface."""

from typing import Any

import pandas as pd
import numpy as np
import xarray as xr

from ...capabilities.base import BoundCapability
from ...capabilities.models import (
    Event,
    UploadMatch,
)
from .._shared import catalog_row, event_value, observation_value, upload_match_value


class Sevir(BoundCapability):
    """
    Sevir: public API for request-bound SEVIR catalog and observation access.

    This wrapper exposes a clean interface to:
      • Read metadata and time axes for the bound event
      • Search catalog events without loading raster pixels
      • Load directly computable VIL observation arrays
      • Match a request-bound upload to a catalog event

    Parameters
    ----------
    recorder : callable
        Runtime callback that records public tool calls and cache hits.
    binding : RequestBinding
        Request identity constructed before the Agent starts. The event or
        upload normally does not need to be passed again to public methods.

    Notes
    -----
    SEVIR is the observation archive in the Agent environment. Use
    :meth:`search` to discover historical candidates and
    :meth:`observations` to load their real observed evolution. CasCast is a
    forecast model and is not a substitute for these historical observations.
    """

    # ------------------------------------------------------------------
    # Event discovery and metadata
    # ------------------------------------------------------------------
    def event(self, subject_key: str | None = None) -> Event:
        """
        Return metadata and the time axis for one SEVIR event.

        Parameters
        ----------
        subject_key : str, optional
            Exact key returned by :meth:`search`. When omitted, use the exact
            subject bound to this request.

        Returns
        -------
        Event
            Object with fields ``subject_key``, ``id``, ``event_anchor_time_utc``, ``event_type``,
            ``frame_times_utc``, ``minute_offsets``, ``south``, ``west``,
            ``north`` and ``east``.

        Examples
        --------
        >>> event = sevir.event()
        >>> event.id
        >>> event.minute_offsets[-1]
        """
        from .Catalog import get_sevir_event_metadata

        value = self._subject_key(subject_key)
        return self._invoke(
            "sevir.event", {"event_id": value}, get_sevir_event_metadata, event_value,
            public_arguments={"subject_key": value},
        )

    def search(
        self,
        query: str = "",
        year: int | None = None,
        month: int | None = None,
        event_type: str = "",
        region: dict[str, float] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_km: float | None = None,
        start_time_utc: str | None = None,
        end_time_utc: str | None = None,
        require_prior_minutes: int | None = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """
        Search SEVIR catalog metadata without loading raster pixels.

        Parameters
        ----------
        query : str, default ""
            Optional free-text catalog query.
        year : int, optional
            Filter by UTC year.
        month : int, optional
            Filter by UTC month, from 1 to 12.
        event_type : str, default ""
            Optional catalog event-type filter.
        region : dict, optional
            WGS84 bounding box with ``south``, ``west``, ``north`` and ``east``.
            Records whose footprint intersects it are returned.
        latitude, longitude : float, optional
            WGS84 centre for a distance search. Supply both or neither.
        radius_km : float, optional
            Maximum geodesic distance from the point to an event footprint.
            Defaults to 100 km when coordinates are supplied.
        start_time_utc, end_time_utc : str, optional
            Inclusive UTC time window for event anchors.
        require_prior_minutes : int, optional
            Require an observation sequence reaching at least this many minutes
            before its event anchor. This is useful for the preceding-hour
            local-observation stage and does not load pixels.
        limit : int, default 10
            Maximum number of records to return.

        Returns
        -------
        pandas.DataFrame
            Rows with stable columns ``subject_key``, ``event_id``, ``event_anchor_time_utc``,
            ``event_type``, ``south``, ``west``, ``north``, ``east``,
            ``distance_km``, ``data_min``, ``data_max``, ``missing_fraction``
            ``observation_status`` and explicit preceding-history continuity
            fields. ``subject_key`` identifies an exact catalog row even when
            event IDs repeat. Query metadata is in ``attrs`` under
            ``criteria``, ``total_matches`` and ``returned_count``.

        Examples
        --------
        >>> result = sevir.search(region={"south": 45, "west": -105, "north": 50, "east": -96},
        ...                       start_time_utc="2019-06-27T00:00:00Z")
        >>> first_subject = result.iloc[0]["subject_key"]
        """
        from .Catalog import search_sevir_catalog

        arguments = {
            "query": query,
            "year": year,
            "month": month,
            "event_type": event_type,
            "region": region,
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "require_prior_minutes": require_prior_minutes,
            "limit": int(limit),
        }
        return self._invoke(
            "sevir.search",
            arguments,
            search_sevir_catalog,
            lambda raw: _search_frame(raw, arguments),
        )

    # ------------------------------------------------------------------
    # Observation access
    # ------------------------------------------------------------------
    def observations(
        self,
        subject_key: str | None = None,
        minutes_before: int = 60,
        minutes_after: int = 0,
    ) -> xr.DataArray:
        """
        Load a real observed VIL sequence with axes time, row and column.

        Parameters
        ----------
        subject_key : str, optional
            Exact catalog key. When omitted, use the request-bound subject.
        minutes_before : int, default 60
            Minutes at or before the event anchor to include.
        minutes_after : int, default 0
            Minutes after the anchor to include. Keep this zero while selecting
            historical analogues so future observations cannot leak in.

        Returns
        -------
        xarray.DataArray
            Named ``vil`` with dimensions ``time``, ``row`` and ``column``.
            Coordinates include ``time_utc`` and ``minute_offset``; missing
            encoded VIL values are represented as NaN and the original code
            255 is recorded in ``attrs["missing_value"]``.

            For historical analogue evaluation, select candidates using their
            observations at minutes <= 0 and reserve minutes > 0 for the later
            evolution being compared. This prevents future observations from
            leaking into analogue selection.

        Examples
        --------
        >>> vil = sevir.observations()
        >>> prior_hour = vil.where((vil.minute_offset >= -60) & (vil.minute_offset <= 0), drop=True)
        >>> peak = float(prior_hour.max(skipna=True))
        """
        from .Catalog import load_sevir_vil_event

        arguments = {
            "event_id": self._subject_key(subject_key),
            "minutes_before": int(minutes_before),
            "minutes_after": int(minutes_after),
        }
        return self._invoke(
            "sevir.observations", arguments, load_sevir_vil_event, observation_value,
            public_arguments={
                "subject_key": arguments["event_id"],
                "minutes_before": int(minutes_before),
                "minutes_after": int(minutes_after),
            },
        )

    def channel_observations(
        self,
        channel: str,
        subject_key: str | None = None,
        minutes_before: int = 60,
        minutes_after: int = 0,
    ) -> xr.DataArray:
        """
        Load satellite or lightning channel observations for a SEVIR event.

        Parameters
        ----------
        channel : str
            Channel type: 'ir069', 'ir107', 'vis', or 'lght'.
        subject_key : str, optional
            Exact key from :meth:`search`. When omitted, use the bound event.
        minutes_before : int, default 60
            How many minutes of history before the event anchor to load.
        minutes_after : int, default 0
            How many minutes after the event anchor to load.

        Returns
        -------
        xarray.DataArray
            Observed channel evolution with dimensions (time, row, column).
            Coordinates include minute_offset and time_utc.

        Examples
        --------
        >>> ir107 = sevir.channel_observations("ir107")
        >>> lght = sevir.channel_observations("lght", minutes_before=30)
        """
        from .MultiChannel import load_sevir_channel_event

        arguments = {
            "event_id": self._subject_key(subject_key),
            "channel": channel,
            "minutes_before": int(minutes_before),
            "minutes_after": int(minutes_after),
        }
        return self._invoke(
            "sevir.channel_observations", arguments, load_sevir_channel_event, observation_value,
            public_arguments={
                "subject_key": arguments["event_id"],
                "channel": channel,
                "minutes_before": int(minutes_before),
                "minutes_after": int(minutes_after),
            },
        )

    def multimodal_observations(
        self,
        channels: list[str] | None = None,
        subject_key: str | None = None,
        minutes_before: int = 60,
        minutes_after: int = 0,
    ) -> dict[str, xr.DataArray]:
        """
        Load multiple channels simultaneously for multi-modal analysis.

        Parameters
        ----------
        channels : list of str, optional
            Channels to load. Defaults to ['vil', 'ir107', 'lght'].
        subject_key : str, optional
            Exact key from :meth:`search`. When omitted, use the bound event.
        minutes_before : int, default 60
            How many minutes of history before the event anchor to load.
        minutes_after : int, default 0
            How many minutes after the event anchor to load.

        Returns
        -------
        dict[str, xarray.DataArray]
            Dictionary mapping channel names to their DataArrays.

        Examples
        --------
        >>> data = sevir.multimodal_observations(["vil", "ir107", "lght"])
        >>> vil_peak = data["vil"].max(skipna=True)
        >>> ir_min = data["ir107"].min(skipna=True)
        >>> lght_count = (data["lght"] > 0).sum()
        """
        from .MultiChannel import load_sevir_multimodal_event

        if channels is None:
            channels = ["vil", "ir107", "lght"]

        arguments = {
            "event_id": self._subject_key(subject_key),
            "channels": channels,
            "minutes_before": int(minutes_before),
            "minutes_after": int(minutes_after),
        }
        return self._invoke(
            "sevir.multimodal_observations",
            arguments,
            load_sevir_multimodal_event,
            lambda raw: _multimodal_observation_dict(raw),
            public_arguments={
                "subject_key": arguments["event_id"],
                "channels": channels,
                "minutes_before": int(minutes_before),
                "minutes_after": int(minutes_after),
            },
        )

    def observation_batch(
        self,
        subject_keys: list[str] | tuple[str, ...],
        minutes_before: int = 60,
        minutes_after: int = 0,
    ) -> xr.Dataset:
        """Load several historical observation sequences in one tool call.

        Parameters
        ----------
        subject_keys : list or tuple of str
            Exact candidate keys returned by :meth:`search`, limited to 20.
        minutes_before, minutes_after : int
            Time window relative to each event anchor, both inclusive.

        Returns
        -------
        xarray.Dataset
            Data variable ``vil(event, time, row, column)``. Event IDs,
            valid times and minute offsets are coordinates. Requested,
            loaded and unavailable IDs are stored in dataset attributes.

        Examples
        --------
        >>> found = sevir.search(year=2019, limit=8)
        >>> batch = sevir.observation_batch(found["subject_key"].tolist())
        >>> peaks = batch["vil"].max(dim=("time", "row", "column"), skipna=True)
        """
        from .Catalog import load_sevir_vil_events

        identifiers = tuple(dict.fromkeys(
            str(value).strip().lower() for value in subject_keys if str(value).strip()
        ))
        if not identifiers:
            raise ValueError("subject_keys must contain at least one exact SEVIR subject key")
        if len(identifiers) > 20:
            raise ValueError("subject_keys is limited to 20 events per batch")
        if int(minutes_before) < 0 or int(minutes_after) < 0:
            raise ValueError("minutes_before and minutes_after must be non-negative")
        arguments = {
            "event_ids": identifiers,
            "minutes_before": int(minutes_before),
            "minutes_after": int(minutes_after),
        }
        return self._invoke(
            "sevir.observation_batch",
            arguments,
            load_sevir_vil_events,
            lambda raw: _observation_batch_dataset(raw, identifiers),
            public_arguments={
                "subject_keys": identifiers,
                "minutes_before": int(minutes_before),
                "minutes_after": int(minutes_after),
            },
        )

    def observation_summaries(
        self,
        subject_keys: list[str] | tuple[str, ...],
        minutes_before: int = 60,
        minutes_after: int = 0,
        threshold: float = 133.0,
    ) -> pd.DataFrame:
        """Return comparable per-event observation facts without exposing pixels.

        Parameters
        ----------
        subject_keys : list or tuple of str
            Exact candidate keys returned by :meth:`search`, limited to 20.
        minutes_before, minutes_after : int
            Inclusive window relative to each event anchor.
        threshold : float, default 133.0
            Encoded VIL threshold used for active-area and centroid summaries.

        Returns
        -------
        pandas.DataFrame
            One row per requested event with stable columns ``subject_key``,
            ``event_id``, ``event_anchor_time_utc``, ``frame_count``,
            ``peak_vil``, ``mean_vil``, ``active_area_first``,
            ``active_area_last``, ``centroid_row_first``,
            ``centroid_column_first``, ``centroid_row_last``,
            ``centroid_column_last``, ``motion_row``, ``motion_column``,
            ``motion_direction``, ``continuity`` and ``data_status``.
            Missing events remain in the table with ``data_status`` set to
            ``unavailable``. The method computes measurements only; it does
            not select an analogue or make a weather conclusion.

        Examples
        --------
        >>> found = sevir.search(region=region, limit=12)
        >>> summary = sevir.observation_summaries(
        ...     found["subject_key"].tolist(), minutes_before=60)
        >>> summary.sort_values("peak_vil").tail(3)
        """
        from .Catalog import load_sevir_vil_events

        identifiers = tuple(dict.fromkeys(
            str(value).strip().lower() for value in subject_keys if str(value).strip()
        ))
        if not identifiers:
            raise ValueError("subject_keys must contain at least one exact SEVIR subject key")
        if len(identifiers) > 20:
            raise ValueError("subject_keys is limited to 20 events per batch")
        if int(minutes_before) < 0 or int(minutes_after) < 0:
            raise ValueError("minutes_before and minutes_after must be non-negative")
        if not np.isfinite(float(threshold)):
            raise ValueError("threshold must be finite")
        arguments = {
            "event_ids": identifiers,
            "minutes_before": int(minutes_before),
            "minutes_after": int(minutes_after),
            "threshold": float(threshold),
        }
        return self._invoke(
            "sevir.observation_summaries",
            arguments,
            load_sevir_vil_events,
            lambda raw: _observation_summary_frame(
                raw, identifiers, float(threshold), arguments
            ),
            public_arguments={
                "subject_keys": identifiers,
                "minutes_before": int(minutes_before),
                "minutes_after": int(minutes_after),
                "threshold": float(threshold),
            },
        )

    def match_upload(self, upload_id: str | None = None) -> UploadMatch:
        """
        Match a request-bound upload to a verified catalog event when possible.

        Parameters
        ----------
        upload_id : str, optional
            Explicit upload identifier. When omitted, use the request-bound upload.

        Returns
        -------
        UploadMatch
            Object with fields ``upload_id``, ``subject_key``, ``event_id``, ``confidence`` and
            ``method``. ``event_id`` is None when no unique match exists.

        Examples
        --------
        >>> match = sevir.match_upload()
        >>> event = sevir.event(match.subject_key) if match.subject_key else None
        """
        from ...data_access.uploads import match_uploaded_event

        value = self._upload_id(upload_id)

        return self._invoke(
            "sevir.match_upload", {"upload_id": value}, match_uploaded_event,
            lambda raw: upload_match_value(value, raw),
        )


def _search_frame(raw: dict[str, Any], arguments: dict[str, Any]) -> pd.DataFrame:
    """Build the standard catalog table and preserve query metadata in attrs.

    Parameters
    ----------
    raw : dict
        Backend search payload.
    arguments : dict
        Canonical search arguments.
    Returns
    -------
    pandas.DataFrame
        Canonical catalog table.
    """

    frame = pd.DataFrame([catalog_row(item) for item in raw.get("records", [])])
    columns = ["subject_key", "event_id", "event_anchor_time_utc", "event_type", "south", "west", "north", "east",
               "distance_km", "data_min", "data_max", "missing_fraction", "observation_status",
               "history_span_minutes", "history_expected_frames", "history_available_frames",
               "history_is_continuous"]
    frame = frame.reindex(columns=columns)
    frame.attrs.update({
        "criteria": dict(arguments),
        "total_matches": int(raw.get("total_matches", 0)),
        "returned_count": int(raw.get("returned_count", len(frame))),
    })
    return frame


def _observation_batch_dataset(raw: dict[str, Any], identifiers: tuple[str, ...]) -> xr.Dataset:
    """Combine loaded standard observation arrays into one xarray Dataset.

    Parameters
    ----------
    raw : dict
        Backend batch payload.
    identifiers : tuple of str
        Requested event IDs.
    Returns
    -------
    xarray.Dataset
        Combined VIL observations and availability metadata.
    """

    arrays = [observation_value(item) for item in raw.get("records", [])]
    if not arrays:
        return xr.Dataset(attrs={
            "requested_subject_keys": identifiers,
            "loaded_subject_keys": (),
            "loaded_event_ids": (),
            "unavailable_subject_keys": tuple(
                str(item.get("subject_key") or "").lower()
                for item in raw.get("unavailable", [])
            ),
        })
    ids = tuple(str(item.attrs.get("event_id") or "").upper() for item in arrays)
    keys = tuple(str(item.attrs.get("subject_key") or "").lower() for item in arrays)
    # Different events can legitimately contribute 12 or 13 frames to the
    # same inclusive minute window.  Make the semantic minute offset the
    # dimension index before concatenation so xarray performs an outer
    # alignment and pads unavailable event/minute cells with NaN.
    aligned = [
        item.assign_coords(
            time=("time", np.asarray(item.coords["minute_offset"].values, dtype=np.int32))
        )
        for item in arrays
    ]
    data = xr.concat(aligned, dim="event", join="outer")
    data = data.assign_coords(event_id=("event", np.asarray(ids, dtype=str)))
    data = data.assign_coords(subject_key=("event", np.asarray(keys, dtype=str)))
    data = data.to_dataset(name="vil")
    data.attrs.update({
        "requested_subject_keys": identifiers,
        "loaded_subject_keys": keys,
        "loaded_event_ids": ids,
        "unavailable_subject_keys": tuple(
            str(item.get("subject_key") or "").lower()
            for item in raw.get("unavailable", [])
        ),
    })
    return data


def _observation_summary_frame(
    raw: dict[str, Any],
    identifiers: tuple[str, ...],
    threshold: float,
    arguments: dict[str, Any],
) -> pd.DataFrame:
    """Compute bounded comparable facts from a backend observation batch.

    Parameters
    ----------
    raw : dict
        Backend batch payload containing loaded records and unavailable entries.
    identifiers : tuple of str
        Requested exact SEVIR subject keys.
    threshold : float
        Encoded VIL threshold for active-area measurements.
    arguments : dict
        Canonical request arguments preserved in the returned frame metadata.

    Returns
    -------
    pandas.DataFrame
        One comparable summary row per requested subject key.
    """

    rows: list[dict[str, Any]] = []
    loaded_by_key = {
        str(item.get("subject_key") or item.get("event_id") or "").strip().lower(): item
        for item in raw.get("records", [])
        if isinstance(item, dict)
    }
    unavailable = {
        str(item.get("subject_key") or item.get("event_id") or "").strip().lower(): str(
            item.get("reason") or item.get("status") or "unavailable"
        )
        for item in raw.get("unavailable", [])
        if isinstance(item, dict)
    }
    for key in identifiers:
        record = loaded_by_key.get(key)
        if record is None:
            rows.append({"subject_key": key, "event_id": "", "data_status": "unavailable",
                         "unavailable_reason": unavailable.get(key, "unavailable")})
            continue
        values = np.asarray(
            record.get("frames", record.get("values", [])), dtype=np.float32
        )
        values = np.where(values == 255, np.nan, values)
        valid = np.isfinite(values)
        valid_values = values[valid]
        row: dict[str, Any] = {
            "subject_key": str(record.get("subject_key") or key).lower(),
            "event_id": str(record.get("event_id") or "").upper(),
            "event_anchor_time_utc": str(record.get("event_anchor_time_utc") or record.get("time_utc") or ""),
            "frame_count": int(values.shape[0]) if values.ndim == 3 else 0,
            "peak_vil": float(np.nanmax(values)) if valid_values.size else np.nan,
            "mean_vil": float(np.nanmean(values)) if valid_values.size else np.nan,
            "active_area_first": np.nan,
            "active_area_last": np.nan,
            "centroid_row_first": np.nan,
            "centroid_column_first": np.nan,
            "centroid_row_last": np.nan,
            "centroid_column_last": np.nan,
            "motion_row": np.nan,
            "motion_column": np.nan,
            "motion_direction": "unknown",
            "continuity": np.nan,
            "data_status": "available",
            "unavailable_reason": "",
        }
        if values.ndim == 3 and values.shape[0]:
            active_areas = np.sum(np.isfinite(values) & (values >= threshold), axis=(1, 2))
            row["active_area_first"] = int(active_areas[0])
            row["active_area_last"] = int(active_areas[-1])
            centroids = []
            for frame in (values[0], values[-1]):
                mask = np.isfinite(frame) & (frame >= threshold)
                if mask.any():
                    yy, xx = np.nonzero(mask)
                    weights = frame[mask].astype(np.float64)
                    centroids.append((float(np.average(yy, weights=weights)),
                                      float(np.average(xx, weights=weights))))
                else:
                    centroids.append((np.nan, np.nan))
            (r0, c0), (r1, c1) = centroids
            row.update({"centroid_row_first": r0, "centroid_column_first": c0,
                        "centroid_row_last": r1, "centroid_column_last": c1,
                        "motion_row": r1 - r0 if np.isfinite(r0) and np.isfinite(r1) else np.nan,
                        "motion_column": c1 - c0 if np.isfinite(c0) and np.isfinite(c1) else np.nan})
            dr, dc = row["motion_row"], row["motion_column"]
            if np.isfinite(dr) and np.isfinite(dc) and (abs(dr) + abs(dc) > 0):
                vertical = "south" if dr > 0 else "north" if dr < 0 else ""
                horizontal = "east" if dc > 0 else "west" if dc < 0 else ""
                row["motion_direction"] = f"{vertical}-{horizontal}".strip("-")
            expected = max(1, int(round((arguments["minutes_before"] + arguments["minutes_after"]) / 5)) + 1)
            row["continuity"] = min(1.0, float(row["frame_count"]) / expected)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs.update({"criteria": dict(arguments), "threshold": threshold,
                        "requested_count": len(identifiers),
                        "returned_count": len(frame)})
    return frame


def _multimodal_observation_dict(raw: dict[str, Any]) -> dict[str, xr.DataArray]:
    """Convert multimodal backend payload to dictionary of DataArrays.

    Parameters
    ----------
    raw : dict
        Backend multimodal payload with channel_data dict.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary mapping channel names to their observation DataArrays.
    """
    result = {}
    channel_data = raw.get("channel_data", {})

    for channel, channel_raw in channel_data.items():
        if channel_raw.get("status") == "success":
            result[channel] = observation_value(channel_raw)

    return result
