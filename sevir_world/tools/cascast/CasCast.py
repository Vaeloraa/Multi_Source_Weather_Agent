from __future__ import annotations

"""Public CasCast tool for request-bound local ensemble nowcasting."""

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import xarray as xr

from ...capabilities.base import BoundCapability
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

    from ..sevir.Catalog import get_sevir_event_metadata

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
    truth = raw.get("ground_truth") if isinstance(raw.get("ground_truth"), dict) else {}
    truth_values = np.asarray(truth.get("frames", []), dtype=np.float32)
    if truth.get("available") and truth_values.size:
        truth_values = truth_values.copy()
        truth_values[truth_values == float(truth.get("missing_value", 255))] = np.nan
        dataset["ground_truth_vil"] = (("lead_minute", "row", "column"), truth_values)
    return dataset


class CasCast(BoundCapability):
    """Request-bound local ensemble nowcasting with a standard xarray result.

    The bound request selects either an event or an upload. The Agent does not
    choose between separate event and upload methods.
    """

    def forecast(
        self,
        horizon_hours: float = 1.0,
        ensemble_members: int = 10,
        include_ground_truth: bool = False,
    ) -> xr.Dataset:
        """Return the bound subject's CasCast forecast as an xarray Dataset.

        Parameters
        ----------
        horizon_hours : float, default 1.0
            Forecast horizon: 1.0, 2.0 or 3.0 hours.
        ensemble_members : int, default 10
            Number of cached ensemble members to request.
        include_ground_truth : bool, default False
            Include later observations for retrospective verification.

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
            and ``rolling_segments``. A 2–3 hour observation-reset product is
            therefore never presented as a free-running forecast.

        Examples
        --------
        >>> forecast = cascast.forecast()
        >>> spread = forecast["forecast_member_vil"].std(dim="member")
        >>> peak = forecast["forecast_mean_vil"].max(dim=("row", "column"))
        """
        horizon = float(horizon_hours)
        if horizon not in {1.0, 2.0, 3.0}:
            raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
        if self.request.assets.upload_id:
            from .Forecast import get_uploaded_vil_sequence_forecast

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
        from .Forecast import get_rolling_forecast_raw_values

        subject_key = self._subject_key(None)
        from ..sevir.Catalog import get_sevir_event_metadata

        exact = get_sevir_event_metadata(subject_key)["record"]
        event_id = str(exact["id"]).upper()
        arguments = {
            "event_id": subject_key, "horizon_hours": horizon,
            "model_name": "cascast_live", "ensemble_members": int(ensemble_members),
            "include_ground_truth": bool(include_ground_truth), "array_transport": True,
        }
        return self._invoke(
            "cascast.forecast", arguments, get_rolling_forecast_raw_values,
            lambda raw: _forecast_dataset(raw, _event_anchor_time(subject_key), subject_key),
            public_arguments={
                "horizon_hours": horizon,
                "ensemble_members": int(ensemble_members),
                "include_ground_truth": bool(include_ground_truth),
            },
        )
