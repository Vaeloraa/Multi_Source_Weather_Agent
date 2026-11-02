"""Public SEVIR tool and its request-bound catalog/observation interface."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import pandas as pd
import numpy as np
import xarray as xr

from ...capabilities.base import BoundCapability
from ...capabilities.contracts import ParameterSpec, tool_parameters
from .channels import SUPPORTED_CHANNELS
from ...capabilities.dataframe_schema import (
    SEVIR_ANALOGUE,
    SEVIR_CATALOG_VIL,
    normalize_dataframe,
)
from ...capabilities.models import (
    Event,
    UploadMatch,
)
from .._shared import (
    catalog_row, event_value, observation_value, require_utc_text,
    upload_match_value,
)
from ..geospatial.contracts import BoundingBox, GeoPoint


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

    def _channel_event_id(self, explicit: str | None = None) -> str:
        """Resolve a VIL subject key back to its shared event ID for channels.

        Satellite channels have their own catalog subject keys, so passing the
        VIL ``sevir:...`` key directly to ``_catalog_event_by_type`` can never
        find VIS/IR/LGHT. Channel loaders must therefore use the event ID.

        Parameters
        ----------
        explicit : str, optional
            Event ID or VIL subject key supplied by the caller.

        Returns
        -------
        str
            Canonical uppercase SEVIR event ID.
        """

        candidate = str(
            explicit
            or self.request.assets.event_id
            or self.request.assets.sevir_subject_key
            or ""
        ).strip()
        if candidate.lower().startswith("sevir:"):
            from ...tools.sevir.repository import _catalog_subject_index

            record = _catalog_subject_index().get(candidate.lower())
            candidate = str(record.get("id") or "") if record else ""
        return self._event_id(candidate)

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
        from ...tools.sevir.repository import get_sevir_event_metadata

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
        region: Any | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_km: float | None = None,
        start_time_utc: str | None = None,
        end_time_utc: str | None = None,
        require_prior_minutes: int | None = None,
        limit: int = 10,
        *,
        observation_time_utc: str | None = None,
        time_match: str = "exact",
        tolerance_seconds: float | None = None,
    ) -> pd.DataFrame:
        """
        Search SEVIR catalog metadata without loading raster pixels.

        Parameters
        ----------
        query : str, default ""
            Optional free-text catalog query.
        year : int, optional
            Filter by event-anchor UTC year (not the frame year).
        month : int, optional
            Filter by event-anchor UTC month, from 1 to 12.
        event_type : str, default ""
            Optional catalog event-type filter.
        region : Any, optional
            Region, WGS84 bounding box or GeoJSON polygon. Records whose
            catalog footprint intersects its bounding box are returned.
        latitude, longitude : float, optional
            WGS84 centre for a distance search. Supply both or neither.
        radius_km : float, optional
            Maximum geodesic distance from the point to an event footprint.
            Defaults to 100 km when coordinates are supplied.
        start_time_utc, end_time_utc : str, optional
            Inclusive time window for event anchors. ISO-8601 timestamps must
            include an explicit timezone: Z for UTC (for example,
            "2017-09-12T00:00:00Z") or an offset such as +00:00.
            Resolve the user's local timezone before converting local dates;
            timestamps without a timezone are rejected.
        require_prior_minutes : int, optional
            Require an observation sequence reaching at least this many minutes
            before its event anchor. This is useful for the preceding-hour
            local-observation stage and does not load pixels.
        limit : int, default 10
            Maximum number of records to return.
        observation_time_utc : str, optional
            Find an actual observation frame at this absolute ISO-8601 time.
            An explicit timezone (Z or UTC offset) is required; resolve the
            user's local timezone before calling. Do not equate the frame
            time with the event anchor, or add anchor filters unnecessarily.
        time_match : {'exact', 'nearest'}, default 'exact'
            Exact frame matching is the default. Nearest is explicit opt-in.
            All matching records are ranked by frame-time distance before limit.
        tolerance_seconds : float, optional
            Required positive finite bound for nearest; forbidden for exact.

        Returns
        -------
        pandas.DataFrame
            Schema ``sevir.catalog.vil@1.1.0``. Rows have stable columns
            ``subject_key``, ``event_id``, ``event_anchor_time_utc``,
            ``event_type``, ``channel``, ``south``, ``west``, ``north``, ``east``,
            ``distance_km``, ``min_vil``, ``max_vil``, ``missing_fraction``,
            ``observation_status`` and explicit preceding-history continuity
            fields. ``subject_key`` identifies an exact catalog row even when
            event IDs repeat. Query metadata is in ``attrs`` under
            ``criteria``, ``total_matches``, ``returned_count``, ``truncated``
            and ``availability``. Matched frames expose ``observation_time_utc``,
            ``time_match_kind`` and signed ``time_delta_seconds``. Availability
            proves only filtered catalog frame coverage, never readable pixels
            or full regional coverage. Region matching uses bounding boxes;
            pass the actual polygon to observations for pixel masking.

        Examples
        --------
        >>> result = sevir.search(region={"south": 45, "west": -105, "north": 50, "east": -96},
        ...                       start_time_utc="2019-06-27T00:00:00Z")
        >>> first_subject = result.iloc[0]["subject_key"]
        """
        from ...tools.sevir.repository import search_sevir_catalog

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
            "observation_time_utc": observation_time_utc,
            "time_match": time_match,
            "tolerance_seconds": tolerance_seconds,
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
        minutes_before: int | None = None,
        minutes_after: int | None = None,
        *,
        observation_time_utc: str | None = None,
        time_match: str = "exact",
        tolerance_seconds: float | None = None,
        region: Any | None = None,
    ) -> xr.DataArray:
        """
        Load a real observed VIL sequence with axes time, row and column.

        Parameters
        ----------
        subject_key : str, optional
            Exact catalog key. When omitted, use the request-bound subject.
        minutes_before : int, optional
            Minutes before the event anchor. Defaults to 60 in relative mode.
        minutes_after : int, optional
            Minutes after the anchor to include. Keep this zero while selecting
            historical analogues so future observations cannot leak in.
            Defaults to 0 in relative mode. Relative windows cannot be mixed
            with observation_time_utc.
        observation_time_utc : str, optional
            Absolute ISO-8601 target with explicit timezone. Selects one actual
            frame directly, including positive offsets after the event anchor.
        time_match : {'exact', 'nearest'}, default 'exact'
            Never silently substitute a different frame for the target.
        tolerance_seconds : float, optional
            Required positive finite bound when nearest is explicitly selected.
        region : Any, optional
            Region, WGS84 bbox or GeoJSON polygon. Pixels outside the region
            become NaN. The original grid is retained for correct geolocation.

        Returns
        -------
        xarray.DataArray
            Named ``vil`` with dimensions ``time``, ``row`` and ``column``.
            ``time`` is a datetime64 UTC index, with ``time_utc`` (ISO Z text)
            and ``minute_offset`` coordinates. Units are ``encoded_vil``;
            missing encoded VIL values are represented as NaN and the original code
            255 is recorded in ``attrs["missing_value"]``.
            ``attrs["time_selection"]`` identifies the requested and actual
            frame; ``attrs["spatial_selection"]`` reports the masked extent
            and regional coverage. An intersecting tile does not imply that
            the whole requested region has observations. Read failures cannot
            prove regional absence. Unmatched times fail before pixel I/O.

            For historical analogue evaluation, select candidates using their
            observations at minutes <= 0 and reserve minutes > 0 for the later
            evolution being compared. This prevents future observations from
            leaking into analogue selection.

        Examples
        --------
        >>> vil = sevir.observations()
        >>> prior_hour = vil.where((vil.minute_offset >= -60) & (vil.minute_offset <= 0), drop=True)
        >>> peak = float(prior_hour.max(skipna=True))
        >>> frame = sevir.observations(subject_key, observation_time_utc="2018-07-20T02:00:00Z", region=region)
        """
        from ...tools.sevir.repository import load_sevir_vil_event

        subject_value = self._subject_key(subject_key)
        if observation_time_utc is None:
            minutes_before = 60 if minutes_before is None else minutes_before
            minutes_after = 0 if minutes_after is None else minutes_after
        arguments = {
            "event_id": subject_value,
            "minutes_before": int(minutes_before) if minutes_before is not None else None,
            "minutes_after": int(minutes_after) if minutes_after is not None else None,
            "observation_time_utc": observation_time_utc,
            "time_match": time_match,
            "tolerance_seconds": tolerance_seconds,
            "region": region,
        }
        return self._invoke(
            "sevir.observations", arguments, load_sevir_vil_event, observation_value,
            public_arguments={
                "subject_key": subject_value,
                **{key: value for key, value in arguments.items() if key != "event_id"},
            },
        )

    @tool_parameters(channel=ParameterSpec(allowed=SUPPORTED_CHANNELS))
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
        channel : {'ir069', 'ir107', 'vis', 'lght'}
            SEVIR channel code. ``ir069`` and ``ir107`` are infrared channels,
            ``vis`` is visible reflectance, and ``lght`` is lightning
            flash-extent density.
        subject_key : str, optional
            Exact key from :meth:`search`. When omitted, use the bound event.
        minutes_before : int, default 60
            How many minutes of history before the event anchor to load.
        minutes_after : int, default 0
            How many minutes after the event anchor to load.

        Returns
        -------
        xarray.DataArray
            Observed channel evolution with dimensions ``time``, ``row`` and
            ``column``. ``time`` is a datetime64 UTC index, accompanied by
            ``minute_offset`` and ISO-8601 ``time_utc`` text.
            ``attrs["channel"]`` and ``attrs["units"]`` identify the product:
            IR channels are scaled brightness temperature, VIS is scaled
            reflectance, and LGHT is flash-extent density. Missing source
            values are converted to NaN.

        Raises
        ------
        ValueError
            If ``channel`` is not one of ``ir069``, ``ir107``, ``vis`` or
            ``lght``, the time window contains no frames, or the returned time
            axis is inconsistent. Data-source unavailability is reported as a
            failed capability call rather than an empty scientific array.

        Examples
        --------
        >>> ir107 = sevir.channel_observations("ir107")
        >>> lght = sevir.channel_observations("lght", minutes_before=30)
        """
        from .MultiChannel import load_sevir_channel_event

        subject_value = self._subject_key(subject_key)
        arguments = {
            "event_id": self._channel_event_id(subject_key),
            "channel": channel,
            "minutes_before": int(minutes_before),
            "minutes_after": int(minutes_after),
        }
        return self._invoke(
            "sevir.channel_observations", arguments, load_sevir_channel_event,
            lambda raw: observation_value(raw, channel=channel),
            public_arguments={
                "subject_key": subject_value,
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
    ) -> MultimodalObservations:
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
        MultimodalObservations
            Mapping from every available channel name to its DataArray.
            ``available_channels`` lists completed reads and
            ``unavailable_channels`` maps every missing requested channel to
            its exact reason, so partial data still produces an honest figure.
            Each DataArray carries channel-specific ``units`` and missing-value
            metadata; do not interpret IR, VIS or LGHT values as encoded VIL.

        Raises
        ------
        ValueError
            If a requested channel name is unsupported or an available channel
            payload has an invalid time-row-column shape. A missing source for
            one valid channel is recorded in ``unavailable_channels`` and does
            not discard successfully loaded channels.

        Examples
        --------
        >>> data = sevir.multimodal_observations(["vil", "vis", "ir069", "ir107", "lght"], minutes_before=0, minutes_after=0)
        >>> vil_peak = data["vil"].max(skipna=True)
        >>> ir_min = data["ir107"].min(skipna=True)
        >>> missing = data.unavailable_channels
        """
        from .MultiChannel import load_sevir_multimodal_event

        if channels is None:
            channels = ["vil", "ir107", "lght"]

        subject_value = self._subject_key(subject_key)
        arguments = {
            "event_id": self._channel_event_id(subject_key),
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
                "subject_key": subject_value,
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
            Data variable ``vil(event, minute_offset, row, column)``. Events
            align by relative minute, with datetime64 UTC ``time`` and ISO
            ``time_utc`` coordinates on (event, minute_offset). Requested,
            loaded and unavailable IDs are stored in dataset attributes.

        Raises
        ------
        ValueError
            If the subject list is empty, contains more than 20 unique keys,
            or either time-window argument is negative.

        Examples
        --------
        >>> found = sevir.search(year=2019, limit=8)
        >>> batch = sevir.observation_batch(found["subject_key"].tolist())
        >>> peaks = batch["vil"].max(dim=("minute_offset", "row", "column"), skipna=True)
        """
        from ...tools.sevir.repository import load_sevir_vil_events

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


    def find_analogues(
        self,
        top_k: int = 5,
        recall_count: int = 40,
        candidate_subject_keys: list[str] | tuple[str, ...] | None = None,
        same_event_type: bool = False,
        max_month_distance: int | None = None,
        max_geographic_km: float | None = None,
        feature_weights: dict[str, float] | None = None,
        region: dict[str, float] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        query_time_utc: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve past-only, explainable historical VIL analogues.

        This method uses the request-bound event or uploaded 13-frame sequence.
        It discovers the historical corpus from SEVIR Catalog using the explicit
        location/region, loads each exact ``subject_key`` through the repository,
        extracts past-only signatures, and then performs robust-scaled ranking.

        Parameters
        ----------
        top_k : int, default 5
            Number of ranked analogue events to return, from 1 through 100.
        recall_count : int, default 40
            Stage-one candidate count. It must be at least ``top_k``.
        candidate_subject_keys : sequence of str, optional
            Exact frozen candidate set, normally inherited from a prior
            historical discovery step. When omitted, use every spatially and
            temporally eligible Catalog record that passes load/signature QC.
        same_event_type : bool, default False
            Require the catalog event-type label to match the reference.
        max_month_distance : int, optional
            Circular UTC month distance from 0 through 6.
        max_geographic_km : float, optional
            Maximum WGS84 distance from the target/reference centre to a
            Catalog event footprint. Applied during Catalog discovery.
        feature_weights : dict, optional
            Non-negative overrides keyed by exact feature or by one of the
            groups ``intensity``, ``area``, ``temporal``, ``temporal_curve``,
            ``motion``, ``objects``, ``season``, ``geography`` and ``quality``.
        region : dict or BoundingBox, optional
            Explicit WGS84 Catalog bounds, with exactly south, west, north,
            east in degrees. Pass named_region.bbox or
            geospatial.region_bounds(named_region) for Region/GeoJSON inputs.
        latitude, longitude : float, optional
            Explicit WGS84 search point. Supply both or neither.
        query_time_utc : str, optional
            Explicit strict-backtest cutoff; eligible candidates must precede it.

        Returns
        -------
        pandas.DataFrame
            Schema ``sevir.analogue.result@1.0.0``. Ranked rows contain event
            identity, distance, group contributions and largest signed feature
            differences. Smaller ``distance`` means greater similarity under
            the versioned signature and active feature weights; it is not a
            probability or calibrated hazard score. Attributes contain the index
            and signature versions, candidate scope, distance definition,
            provenance, cache identity and scientific limitations.

        Raises
        ------
        ValueError
            If ``top_k`` is outside 1 through 100, ``recall_count`` is smaller
            than ``top_k``, month distance is outside 0 through 6, geographic
            distance is not positive, or a feature weight is negative.
        RuntimeError
            If no versioned signature can be built for the bound subject or
            its observations are unavailable. Fewer eligible candidates than
            ``top_k`` produce fewer rows rather than fabricated matches.

        Examples
        --------
        >>> analogues = sevir.find_analogues(top_k=5, max_month_distance=2)
        >>> analogues[["event_id", "distance", "group_contributions"]]
        """
        from ..retrieval.service import find_bound_analogues

        frozen = tuple(dict.fromkeys(
            str(item).strip().lower()
            for item in (candidate_subject_keys or ())
            if str(item).strip()
        ))
        search_box = BoundingBox.from_value(region)
        if search_box is not None and (latitude is not None or longitude is not None):
            raise ValueError("supply either region or latitude/longitude, not both")
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        search_point = (
            GeoPoint(float(latitude), float(longitude))
            if latitude is not None and longitude is not None else None
        )
        cutoff = (
            require_utc_text(query_time_utc, "query_time_utc")
            if query_time_utc else ""
        )
        arguments = {
            "subject_key": (
                "" if self.request.assets.upload_id else self._subject_key(None)
            ),
            "upload_id": str(self.request.assets.upload_id or ""),
            "top_k": int(top_k),
            "recall_count": int(recall_count),
            "candidate_subject_keys": frozen,
            "same_event_type": bool(same_event_type),
            "max_month_distance": max_month_distance,
            "max_geographic_km": max_geographic_km,
            "feature_weights": dict(feature_weights or {}),
            "region": search_box.as_dict() if search_box else None,
            "latitude": search_point.latitude if search_point else None,
            "longitude": search_point.longitude if search_point else None,
            "query_time_utc": cutoff,
            "allow_remote": self.request.data_policy == "live",
        }

        def to_frame(raw: dict[str, Any]) -> pd.DataFrame:
            """Convert a retrieval result to the public ranked table.

            Parameters
            ----------
            raw : dict
                Versioned analogue-result mapping.

            Returns
            -------
            pandas.DataFrame
                Candidate rows with result metadata in ``attrs``.
            """
            frame = normalize_dataframe(raw.get("records", []), SEVIR_ANALOGUE)
            frame.attrs.update({
                key: raw.get(key) for key in (
                    "schema_version", "query_signature_id", "index_id",
                    "candidate_scope", "distance_definition", "provenance",
                    "elapsed_ms", "warnings", "limitations", "cache_hit",
                    "cache_identity", "index_cache_status",
                )
            })
            return frame

        return self._invoke(
            "sevir.find_analogues",
            arguments,
            find_bound_analogues,
            to_frame,
            public_arguments={
                "top_k": int(top_k),
                "recall_count": int(recall_count),
                "candidate_subject_keys": frozen,
                "same_event_type": bool(same_event_type),
                "max_month_distance": max_month_distance,
                "max_geographic_km": max_geographic_km,
                "feature_weights": dict(feature_weights or {}),
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
        from .uploads import match_uploaded_event

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

    frame = normalize_dataframe(
        [catalog_row(item) for item in raw.get("records", [])],
        SEVIR_CATALOG_VIL,
        attrs={
            "criteria": dict(raw.get("filters", arguments)),
            "total_matches": int(raw.get("total_matches", 0)),
            "returned_count": int(raw.get("returned_count", len(raw.get("records", [])))),
            "truncated": bool(raw.get("truncated", False)),
            "availability": dict(raw.get("availability") or {}),
            "units": raw.get("units", "encoded_vil"),
        },
    )
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
    # same inclusive window. Align by the explicitly named relative axis;
    # never overwrite the absolute UTC coordinate with integer offsets.
    aligned = [item.swap_dims({"time": "minute_offset"}) for item in arrays]
    data = xr.concat(aligned, dim="event", join="outer", coords=["time", "time_utc"], combine_attrs="drop")
    data.coords["time"].attrs["timezone"] = "UTC"
    data.attrs.update(units="encoded_vil", source="SEVIR", missing_value=255)
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
class MultimodalObservations(dict[str, xr.DataArray]):
    """Available channel arrays plus explicit partial-availability metadata."""

    available_channels: tuple[str, ...]
    unavailable_channels: dict[str, str]


def _multimodal_observation_dict(raw: dict[str, Any]) -> MultimodalObservations:
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
    result = MultimodalObservations()
    channel_data = raw.get("channel_data", {})

    for channel, channel_raw in channel_data.items():
        if channel_raw.get("status") in {"ok", "success"}:
            result[channel] = observation_value(channel_raw, channel=channel)

    result.available_channels = tuple(
        str(channel) for channel in raw.get("channels_available", result.keys())
    )
    result.unavailable_channels = {
        str(channel): str(channel_data.get(channel, {}).get("reason") or "data unavailable")
        for channel in raw.get("channels_unavailable", [])
    }

    return result
