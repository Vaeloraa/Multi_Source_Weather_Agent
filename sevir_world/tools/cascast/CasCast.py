"""Public CasCast tool for request-bound local ensemble nowcasting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ...capabilities.base import BoundCapability
from ...capabilities.contracts import ParameterSpec, ProductMetadata, tool_parameters
from ...capabilities.dataframe_schema import (
    CASCAST_AVAILABILITY,
    normalize_dataframe,
)
from .._shared import utc_text


def _forecast_times(initialization_time_utc: str | None, lead_minutes: list[int]) -> tuple[str, ...]:
    """Create UTC valid-time labels for forecast leads.

    Parameters
    ----------
    initialization_time_utc : str or None
        Forecast initialization time.
    lead_minutes : list of int
        Relative lead minutes.
    Returns
    -------
    tuple of str
        ISO-8601 UTC labels.
    """

    if not initialization_time_utc:
        return tuple("" for _ in lead_minutes)
    parsed = datetime.fromisoformat(initialization_time_utc.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return tuple(
        (parsed.astimezone(timezone.utc) + timedelta(minutes=int(lead))).isoformat().replace("+00:00", "Z")
        for lead in lead_minutes
    )


def _event_anchor_time(event_id: str) -> str | None:
    """Read the catalog anchor used for a bound event forecast.

    Parameters
    ----------
    event_id : str
        SEVIR event identity.
    Returns
    -------
    str or None
        Event anchor time.
    """

    from ...tools.sevir.repository import get_sevir_event_metadata

    raw = get_sevir_event_metadata(event_id)
    record = raw.get("record") if isinstance(raw, dict) else None
    return utc_text(record.get("time_utc")) if isinstance(record, dict) else None


def _forecast_dataset(
    raw: dict[str, Any],
    initialization_time_utc: str | None,
    subject_key: str = "",
) -> xr.Dataset:
    """Convert raw CasCast arrays to one standard, directly computable Dataset.

    Parameters
    ----------
    raw : dict
        Backend forecast payload.
    initialization_time_utc : str or None
        Bound initialization time.
    Returns
    -------
    xarray.Dataset
        Named forecast and history variables.
    """

    history = np.asarray(raw.get("history_frames", []), dtype=np.float32)
    mean = np.asarray(raw.get("forecast_frames", []), dtype=np.float32)
    member_records = raw.get("ensemble", {}).get("members", [])
    members = (
        np.stack([np.asarray(item["frames"], dtype=np.float32) for item in member_records])
        if member_records else mean[None, ...]
    )
    if history.size:
        history = history.copy()
        history[history == 255] = np.nan
    history_minutes = [int(item) for item in raw.get("history_relative_minutes", [])]
    lead_minutes = [int(item) for item in (raw.get("forecast_relative_minutes") or raw.get("lead_minutes") or [])]
    initialization = utc_text(initialization_time_utc or raw.get("initialization_time_utc")) or None
    rows, columns = mean.shape[-2:] if mean.ndim >= 2 else (0, 0)
    coords: dict[str, Any] = {
        "history_minute": np.asarray(history_minutes, dtype=np.int32),
        "history_time_utc": ("history_minute", np.asarray(_forecast_times(initialization, history_minutes), dtype=str)),
        "lead_minute": np.asarray(lead_minutes, dtype=np.int32),
        "valid_time_utc": ("lead_minute", np.asarray(_forecast_times(initialization, lead_minutes), dtype=str)),
        "member": np.arange(members.shape[0], dtype=np.int32),
        "member_seed": ("member", np.asarray([int(item.get("seed", index + 1)) for index, item in enumerate(member_records)] if member_records else [1], dtype=np.int64)),
        "row": np.arange(rows, dtype=np.int32),
        "column": np.arange(columns, dtype=np.int32),
    }
    dataset = xr.Dataset(
        data_vars={
            "history_vil": (("history_minute", "row", "column"), history),
            "forecast_mean_vil": (("lead_minute", "row", "column"), mean),
            "forecast_member_vil": (("member", "lead_minute", "row", "column"), members),
        },
        coords=coords,
        attrs={
            "subject_key": str(subject_key or raw.get("upload_id") or "").lower(),
            "subject_kind": "uploaded_sequence" if raw.get("upload_id") else "sevir_event",
            "event_id": str(raw.get("event_id") or "").upper(),
            "upload_id": str(raw.get("upload_id") or ""),
            "model": str(raw.get("model_name") or "cascast"),
            "initialization_time_utc": initialization,
            "units": "encoded_vil",
            "cache_hit": bool(raw.get("cache_hit", False)),
            "forecast_mode": str(raw.get("forecast_mode") or "standard_single_window"),
            "uses_future_observations": bool(raw.get("uses_future_observations", raw.get("observation_reset", False))),
            "rolling_segments": int(raw.get("rolling_segments") or 1),
            "ground_truth_requested": bool(raw.get("ground_truth_requested", False)),
            "ground_truth_available": bool(raw.get("ground_truth_available", False)),
        },
    )
    event_id = str(raw.get("event_id") or "").strip()
    if event_id:
        from ..sevir.repository import catalog_vil_event, spatial_metadata_for_record

        dataset.attrs["spatial_metadata"] = spatial_metadata_for_record(
            catalog_vil_event(subject_key if subject_key and not subject_key.startswith("upload:") else event_id)
        )
    truth = raw.get("ground_truth") if isinstance(raw.get("ground_truth"), dict) else {}
    truth_values = np.asarray(truth.get("frames", []), dtype=np.float32)
    if truth.get("available") and truth_values.size:
        truth_values = truth_values.copy()
        truth_values[truth_values == float(truth.get("missing_value", 255))] = np.nan
        dataset["ground_truth_vil"] = (("lead_minute", "row", "column"), truth_values)
    dataset.attrs["model_id"] = "cascast"
    dataset.attrs["model_version"] = str(raw.get("model_version") or "")
    variables = {}
    for name, field in dataset.data_vars.items():
        role = "observation" if name == "history_vil" else "ground_truth" if name == "ground_truth_vil" else "forecast"
        field.attrs.update(units="encoded_vil", product_role=role)
        times = coords["history_time_utc"][1] if role == "observation" else coords["valid_time_utc"][1]
        variables[name] = {
            "role": role, "units": "encoded_vil", "dimensions": list(field.dims),
            "valid_time_utc": times.tolist(),
        }
    limitations = []
    if dataset.attrs["uses_future_observations"]:
        limitations.append("Uses future observations between segments; retrospective observation-reset product, not a free-running forecast.")
    dataset.attrs["product_metadata"] = ProductMetadata(
        model_id="cascast", model_version=dataset.attrs["model_version"],
        input_identity={
            "subject_key": dataset.attrs["subject_key"], "event_id": dataset.attrs["event_id"],
            "upload_id": dataset.attrs["upload_id"],
        },
        initialization_time_utc=initialization or "", variables=variables,
        cache_identity={key: raw[key] for key in (
            "request_key", "cache_request_key", "forecast_evidence_id", "prediction_source",
        ) if raw.get(key)}, limitations=tuple(limitations),
    ).to_dict()
    dataset.attrs["product_metadata"]["forecast_mode"] = dataset.attrs["forecast_mode"]
    dataset.attrs["product_metadata"]["uses_future_observations"] = dataset.attrs["uses_future_observations"]
    return dataset


class CasCast(BoundCapability):
    """Request-bound local ensemble nowcasting with a standard xarray result.

    An explicit subject can select a catalog event discovered this turn when
    the request is unbound. Existing event/upload bindings remain authoritative.
    """

    def _forecast_subject(self, explicit: str | None) -> str:
        from ..sevir.repository import resolve_sevir_subject_key
        if explicit is not None:
            if self.request.assets.upload_id:
                raise ValueError("explicit catalog subject conflicts with the bound upload")
            selected = resolve_sevir_subject_key(explicit)
            if self.request.assets.event_id or self.request.assets.sevir_subject_key:
                bound = resolve_sevir_subject_key(self._subject_key(None))
                if bound != selected:
                    raise ValueError("explicit subject conflicts with the request-bound event")
            return selected
        return self._subject_key(None)

    def available_forecasts(self, subject_key: str | None = None) -> pd.DataFrame:
        """List forecast horizons and ensemble sizes actually available for this request.

        Parameters
        ----------
        subject_key : str, optional
            Exact subject selected by sevir.search in this turn. As in forecast,
            a conflict with an already bound event or upload is rejected.

        Returns
        -------
        pandas.DataFrame
            Schema ``cascast.forecast.availability@1.0.0``. Cached
            combinations have one row per horizon/member count with columns
            ``subject_key``, ``upload_id``, ``horizon_hours``,
            ``ensemble_members``, ``ground_truth_available``, ``cache_status``
            and ``source``. In live mode, a cache miss instead returns one
            control row with ``cache_status="miss_live_can_run"`` and
            ``action_required="call cascast.forecast()"``; its horizon and
            member count are intentionally null. In cached-only mode, an empty
            table means no compatible cached result. This method discovers
            availability only and never runs inference.

        Examples
        --------
        >>> available = cascast.available_forecasts()
        >>> available[["horizon_hours", "ensemble_members"]]
        """
        from .config import CASCAST_LIVE_CACHE
        selected_subject = self._forecast_subject(subject_key) if subject_key is not None or not self.request.assets.upload_id else ""
        rows: list[dict[str, Any]] = []
        upload_id = str(self.request.assets.upload_id or "").strip()
        if upload_id:
            paths = sorted(CASCAST_LIVE_CACHE.glob(f"upload_{upload_id}_ens*.npz"))
            for path in paths:
                stem = path.stem
                marker = stem.split("_ens", 1)[-1]
                rolling = 1
                if "_roll" in marker:
                    members_text, rolling_text = marker.split("_roll", 1)
                    rolling = int(rolling_text)
                else:
                    members_text = marker
                try:
                    members = int(members_text)
                except ValueError:
                    continue
                rows.append({"subject_key": "", "upload_id": upload_id,
                             "horizon_hours": float(rolling),
                             "ensemble_members": members,
                             "ground_truth_available": False,
                             "cache_status": "available", "source": str(path)})
        else:
            subject_key = selected_subject
            from ...tools.cascast.backend import _matching_cache
            for segment in (0, 1, 2):
                path = _matching_cache(subject_key, segment, None)
                if path is None:
                    continue
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        members = int(np.asarray(payload["ensemble_members"]).shape[0])
                        has_truth = "ground_truth" in payload
                except (OSError, KeyError, ValueError):
                    continue
                rows.append({"subject_key": subject_key, "upload_id": "",
                             "horizon_hours": float(segment + 1),
                             "ensemble_members": members,
                             "ground_truth_available": has_truth,
                             "cache_status": "available", "source": str(path)})
        # Discovery must communicate the live fallback explicitly.  Otherwise
        # an empty DataFrame is too easy for the Agent to misread as a model
        # outage and it will never call ``forecast`` on a cache miss.
        if not rows and self.request.data_policy == "live":
            rows.append({
                "subject_key": selected_subject,
                "upload_id": upload_id,
                "horizon_hours": None,
                "ensemble_members": None,
                "ground_truth_available": False,
                "cache_status": "miss_live_can_run",
                "source": "local_cascast_inference",
                "action_required": "call cascast.forecast()",
            })
        return self._invoke(
            "cascast.available_forecasts",
            {"subject_key": selected_subject,
             "upload_id": upload_id},
            lambda **_: {"status": "ok", "rows": rows},
            lambda raw: normalize_dataframe(
                raw.get("rows", []), CASCAST_AVAILABILITY
            ),
            public_arguments={"subject_key": selected_subject,
                              "upload_id": upload_id},
        )

    @tool_parameters(
        horizon_hours=ParameterSpec(allowed=(1.0, 2.0, 3.0)),
        ensemble_members=ParameterSpec(minimum=1),
    )
    def forecast(
        self,
        horizon_hours: float = 1.0,
        ensemble_members: int = 10,
        include_ground_truth: bool = False,
        *,
        subject_key: str | None = None,
    ) -> xr.Dataset:
        """Forecast a catalog subject selected this turn, or the bound event/upload.

        Parameters
        ----------
        subject_key : str, optional
            Exact catalog subject_key from sevir.search. A unique catalog event
            ID also resolves to its exact row. Pass this for a subject selected
            in the current turn when the request has no bound asset. An explicit
            selection conflicting with the bound event or upload is rejected.
            Selection never mutates RequestBinding.
        horizon_hours : float, default 1.0
            Forecast horizon: 1.0, 2.0 or 3.0 hours.
        ensemble_members : int, default 10
            Ensemble members to read from an exact cache hit or generate during
            live inference. Must be between 1 and the configured live maximum
            (10 by default).
        include_ground_truth : bool, default False
            Include later observations for retrospective verification when
            available. This introduces future information and must not be used
            to make an operational forecast claim.

        Returns
        -------
        xarray.Dataset
            Variables ``history_vil(history_minute,row,column)``,
            ``forecast_mean_vil(lead_minute,row,column)`` and
            ``forecast_member_vil(member,lead_minute,row,column)``. Optional
            ``ground_truth_vil`` is present when requested and available.
            Coordinates include ``history_time_utc``, ``valid_time_utc``,
            ``history_minute``, ``lead_minute``, ``member``, ``row`` and
            ``column``. Attributes explicitly include ``subject_key``,
            ``subject_kind``, ``forecast_mode``, ``uses_future_observations``
            and ``rolling_segments``. ``product_metadata`` declares model identity,
            variable roles, units, valid times and limitations; history and ground
            truth retain their separate roles. A 2–3 hour observation-reset product is
            therefore never presented as a free-running forecast.

        Raises
        ------
        ValueError
            If ``horizon_hours`` is not 1, 2 or 3, the ensemble count is
            outside the configured range, or the request has neither a bound
            upload nor an exact SEVIR subject.
        RuntimeError
            When cached-only policy has no compatible artifact, or live model
            assets/inference are unavailable. Call :meth:`available_forecasts`
            first and report an unavailable forecast rather than substituting
            persistence or observations as CasCast output.

        Examples
        --------
        >>> forecast = cascast.forecast(subject_key=events.iloc[0]["subject_key"])
        >>> spread = forecast["forecast_member_vil"].std(dim="member")
        >>> peak = forecast["forecast_mean_vil"].max(dim=("row", "column"))
        """
        horizon = float(horizon_hours)
        selected_subject = self._forecast_subject(subject_key) if subject_key is not None or not self.request.assets.upload_id else ""
        if self.request.assets.upload_id:
            from .transport import get_uploaded_vil_sequence_forecast

            arguments = {
                "upload_id": self._upload_id(None), "model_name": "cascast_live",
                "ensemble_members": int(ensemble_members), "horizon_hours": horizon,
                "include_ground_truth": bool(include_ground_truth),
                "array_transport": True, "force_live": self.request.data_policy == "live",
            }
            return self._invoke(
                "cascast.forecast", arguments, get_uploaded_vil_sequence_forecast,
                lambda raw: _forecast_dataset(
                    raw,
                    _event_anchor_time(self.request.assets.sevir_subject_key)
                    if self.request.assets.sevir_subject_key else None,
                    f"upload:{self.request.assets.upload_id}",
                ),
                public_arguments={
                    "horizon_hours": horizon,
                    "ensemble_members": int(ensemble_members),
                    "include_ground_truth": bool(include_ground_truth),
                },
            )
        from .transport import get_rolling_forecast_raw_values

        subject_key = selected_subject
        from ...tools.sevir.repository import get_sevir_event_metadata

        exact = get_sevir_event_metadata(subject_key)["record"]
        event_id = str(exact["id"]).upper()
        arguments = {
            "event_id": subject_key, "horizon_hours": horizon,
            "model_name": "cascast_live", "ensemble_members": int(ensemble_members),
            "include_ground_truth": bool(include_ground_truth), "array_transport": True,
            "force_live": self.request.data_policy == "live",
        }
        def convert_forecast(raw):
            if str(raw.get("event_id") or "").upper() != event_id:
                raise ValueError("CasCast backend event identity differs from the selected catalog subject")
            if raw.get("subject_key") and str(raw["subject_key"]).lower() != subject_key:
                raise ValueError("CasCast backend subject identity differs from the selected catalog subject")
            return _forecast_dataset(raw, _event_anchor_time(subject_key), subject_key)
        return self._invoke(
            "cascast.forecast", arguments, get_rolling_forecast_raw_values,
            convert_forecast,
            public_arguments={
                "subject_key": subject_key,
                "horizon_hours": horizon,
                "ensemble_members": int(ensemble_members),
                "include_ground_truth": bool(include_ground_truth),
            },
        )
