from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..backends import forecast
from ..data_access import ensemble

from .base import WorldComponent

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only in minimal deployments
    linear_sum_assignment = None


def _uploaded_raw_contract(
    upload_id: str,
    lead_minutes: list[int],
    ensemble_members: int = 10,
    horizon_hours: float = 1.0,
    include_ground_truth: bool = False,
    array_transport: bool = True,
    force_live: bool = False,
) -> dict[str, Any]:
    """Load up to twelve leads and attach non-probabilistic evidence semantics."""
    selected = list(dict.fromkeys(int(value) for value in lead_minutes))
    if not selected:
        raise ValueError("lead_minutes must contain at least one lead")
    if len(selected) > 12:
        raise ValueError("uncertainty.uploaded_raw supports at most twelve leads per call")
    chunks = [selected[index:index + 4] for index in range(0, len(selected), 4)]
    parts = [
        forecast.get_uploaded_ensemble_raw_values(
            upload_id=upload_id,
            lead_minutes=chunk,
            ensemble_members=ensemble_members,
            horizon_hours=horizon_hours,
            include_ground_truth=include_ground_truth,
            array_transport=True,
            force_live=force_live,
        )
        for chunk in chunks
    ]
    result = dict(parts[0])
    if len(parts) > 1:
        merged_records: list[dict[str, Any]] = []
        member_count = int(parts[0]["ensemble_members"])
        for member_index in range(member_count):
            record = dict(parts[0]["records"][member_index])
            frames = np.ascontiguousarray(
                np.concatenate(
                    [np.asarray(part["records"][member_index]["frames"]) for part in parts],
                    axis=0,
                ),
                dtype=np.float32,
            )
            frames.setflags(write=False)
            record["lead_minutes"] = selected
            record["frames"] = frames if array_transport else frames.tolist()
            merged_records.append(record)
        result["records"] = merged_records
        result["selected_lead_minutes"] = selected
        result["cache_hit"] = all(bool(part.get("cache_hit")) for part in parts)
        if isinstance(result.get("forecast_evidence"), dict):
            result["forecast_evidence"] = dict(result["forecast_evidence"])
            result["forecast_evidence"]["selected_lead_minutes"] = selected
            result["forecast_evidence"]["cache_hit"] = result["cache_hit"]

        truth_available = bool(include_ground_truth) and all(
            bool(part.get("ground_truth_available")) for part in parts
        )
        if truth_available:
            truth_frames = np.ascontiguousarray(
                np.concatenate(
                    [np.asarray(part["ground_truth"]["frames"], dtype=np.uint8) for part in parts],
                    axis=0,
                ),
                dtype=np.uint8,
            )
            truth_frames.setflags(write=False)
            truth = dict(parts[0]["ground_truth"])
            truth["lead_minutes"] = selected
            truth["frames"] = truth_frames if array_transport else truth_frames.tolist()
            result["ground_truth"] = truth
        elif include_ground_truth:
            result["ground_truth"] = {
                "available": False,
                "lead_minutes": selected,
                "frames": [],
                "reason": "held-out truth was not available for every selected lead",
            }
        result["ground_truth_available"] = truth_available
        result["ground_truth_requested"] = bool(include_ground_truth)

    result["backend_chunk_count"] = len(parts)
    result["capabilities"] = list(dict.fromkeys([
        *result.get("capabilities", []),
        "ensemble_agreement",
    ]))
    result["ensemble_semantics"] = {
        "member_count": int(result.get("ensemble_members", ensemble_members)),
        "member_fraction_meaning": "fraction of configured model members satisfying a generated-code condition",
        "calibrated_probability_available": False,
        "member_agreement_is_observed_probability": False,
        "member_spread_is_forecast_error": False,
        "ground_truth_loaded": bool(result.get("ground_truth_available")),
    }
    result["decision_capabilities"] = {
        "official_warning": False,
        "rainfall_rate": False,
        "road_condition": False,
        "route_safety": False,
        "safe_to_drive": False,
    }
    provenance = dict(result.get("provenance") or {})
    provenance.update({
        "member_support_is_calibrated_probability": False,
        "route_safety_evidence_available": False,
    })
    result["provenance"] = provenance
    result["interpretation_limit"] = (
        "These grids describe disagreement among configured CasCast members, not a "
        "calibrated chance of weather, impact, or route safety. Encoded VIL cannot by "
        "itself establish rainfall rate, road flooding, visibility, hail, an official "
        "warning, or whether someone should drive, wait, or reroute. Give only conditional "
        "general safety guidance and direct users to official weather and traffic sources."
    )
    return result


def _components(frame: np.ndarray, threshold: float, minimum_pixels: int) -> list[dict[str, Any]]:
    mask = np.asarray(frame) >= threshold
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    objects: list[dict[str, Any]] = []
    for start_row, start_column in np.argwhere(mask):
        row = int(start_row)
        column = int(start_column)
        if visited[row, column]:
            continue
        queue = deque([(row, column)])
        visited[row, column] = True
        pixels: list[int] = []
        rows: list[int] = []
        columns: list[int] = []
        while queue:
            current_row, current_column = queue.popleft()
            rows.append(current_row)
            columns.append(current_column)
            pixels.append(current_row * width + current_column)
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
        if len(pixels) < minimum_pixels:
            continue
        pixel_array = np.asarray(sorted(pixels), dtype=np.int32)
        objects.append(
            {
                "pixels": pixel_array,
                "pixel_count": len(pixels),
                "area_ratio": len(pixels) / mask.size,
                "bbox": [min(columns), min(rows), max(columns), max(rows)],
                "centroid": [float(np.mean(columns)), float(np.mean(rows))],
                "maximum_vil": float(np.asarray(frame).ravel()[pixel_array].max()),
            }
        )
    objects.sort(key=lambda item: item["pixel_count"], reverse=True)
    return objects[:30]


def _object_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    intersection = np.intersect1d(first["pixels"], second["pixels"], assume_unique=True).size
    union = first["pixel_count"] + second["pixel_count"] - intersection
    return float(intersection / union) if union else 0.0


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if not cost.size:
        return []
    if linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), columns.tolist()))
    candidates = sorted(
        (float(cost[row, column]), row, column)
        for row in range(cost.shape[0])
        for column in range(cost.shape[1])
    )
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    matches = []
    for _, row, column in candidates:
        if row not in used_rows and column not in used_columns:
            matches.append((row, column))
            used_rows.add(row)
            used_columns.add(column)
    return matches


def track_fields(
    fields: np.ndarray,
    threshold: float,
    minimum_region_pixels: int = 1,
    maximum_gap_frames: int = 1,
    pixel_size_km: float | None = None,
    calibration_source: str = "",
) -> dict[str, Any]:
    """Track masks in grid coordinates, with optional explicit calibration.

    Physical distance and area fields are emitted only when the caller supplies
    both a positive ``pixel_size_km`` and a non-empty ``calibration_source``.
    """
    calibrated = pixel_size_km is not None
    if calibrated:
        pixel_size_km = float(pixel_size_km)
        if pixel_size_km <= 0:
            raise ValueError("pixel_size_km must be positive when supplied")
        if not str(calibration_source or "").strip():
            raise ValueError(
                "calibration_source is required when converting pixels to physical units"
            )
    sequence = np.asarray(fields, dtype=np.float32)
    tracks: list[dict[str, Any]] = []
    frame_counts: list[int] = []
    split_events: list[dict[str, Any]] = []
    merge_events: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(sequence):
        objects = _components(frame, threshold, minimum_region_pixels)
        frame_counts.append(len(objects))
        active = [
            index
            for index, track in enumerate(tracks)
            if frame_index - track["frames"][-1] <= maximum_gap_frames + 1
        ]
        cost = np.full((len(active), len(objects)), 1e6, dtype=np.float64)
        ious = np.zeros_like(cost)
        for row, track_index in enumerate(active):
            previous = tracks[track_index]["objects"][-1]
            gap = frame_index - tracks[track_index]["frames"][-1]
            for column, current in enumerate(objects):
                iou = _object_iou(previous, current)
                distance = float(
                    np.linalg.norm(np.asarray(previous["centroid"]) - np.asarray(current["centroid"]))
                )
                ious[row, column] = iou
                if iou >= 0.02 or distance <= 32.0 * gap:
                    cost[row, column] = (1.0 - iou) + distance / (64.0 * gap)
        matched_objects: set[int] = set()
        for row, column in _assign(cost):
            if cost[row, column] >= 1e5:
                continue
            track_index = active[row]
            tracks[track_index]["frames"].append(frame_index)
            tracks[track_index]["objects"].append(objects[column])
            matched_objects.add(column)
        for object_index, current in enumerate(objects):
            if object_index not in matched_objects:
                tracks.append({"frames": [frame_index], "objects": [current]})

        for column in range(len(objects)):
            parents = [active[row] for row in range(len(active)) if ious[row, column] >= 0.10]
            if len(parents) > 1:
                merge_events.append(
                    {"frame_index": frame_index, "parent_track_indices": parents, "object_index": column}
                )
        for row, track_index in enumerate(active):
            children = [column for column in range(len(objects)) if ious[row, column] >= 0.10]
            if len(children) > 1:
                split_events.append(
                    {"frame_index": frame_index, "parent_track_index": track_index, "object_indices": children}
                )

    public_tracks = []
    for index, track in enumerate(tracks, start=1):
        frames = track["frames"]
        objects = track["objects"]
        centroids = np.asarray([item["centroid"] for item in objects], dtype=np.float64)
        segment_distance_pixels = (
            np.linalg.norm(np.diff(centroids, axis=0), axis=1)
            if len(centroids) > 1
            else np.asarray([], dtype=np.float64)
        )
        elapsed_hours = max((frames[-1] - frames[0]) * 5 / 60.0, 0.0)
        path_pixels = float(segment_distance_pixels.sum())
        observations = [
            {
                "frame_index": frame_index,
                "lead_minutes": (frame_index + 1) * 5,
                "centroid": [round(value, 3) for value in item["centroid"]],
                "pixel_count": item["pixel_count"],
                "maximum_vil": round(item["maximum_vil"], 3),
            }
            for frame_index, item in zip(frames, objects)
        ]
        if calibrated:
            for observation in observations:
                observation["area_km2"] = round(
                    observation["pixel_count"] * float(pixel_size_km) ** 2, 3
                )
        public_track = {
                "track_id": f"T{index}",
                "start_frame": frames[0],
                "end_frame": frames[-1],
                "duration_minutes": max(0, (frames[-1] - frames[0]) * 5),
                "duration_definition": "elapsed span from first to last matched frame",
                "left_censored": frames[0] == 0,
                "right_censored": frames[-1] == int(sequence.shape[0]) - 1,
                "complete_lifecycle_available": False,
                "maximum_gap_frames": max(np.diff(frames), default=1) - 1,
                "maximum_pixel_count": max(item["pixel_count"] for item in objects),
                "path_length_pixels": round(path_pixels, 3),
                "mean_speed_pixels_per_hour": (
                    round(path_pixels / elapsed_hours, 3) if elapsed_hours else 0.0
                ),
                "mean_centroid": [round(value, 3) for value in centroids.mean(axis=0)],
                "observations": observations,
            }
        if calibrated:
            path_km = path_pixels * float(pixel_size_km)
            public_track.update({
                "path_length_km": round(path_km, 3),
                "mean_speed_kmh": round(path_km / elapsed_hours, 3) if elapsed_hours else 0.0,
            })
        public_tracks.append(public_track)
    public_tracks.sort(
        key=lambda item: (item["duration_minutes"], item["maximum_pixel_count"]), reverse=True
    )
    return {
        "capabilities": ["grid_object_tracking", "split_merge_diagnostics"]
        + (["calibrated_distance", "calibrated_area"] if calibrated else []),
        "provenance": {
            "spatial_units": "kilometres and square kilometres" if calibrated else "pixels",
            "physical_calibration_available": bool(calibrated),
            "calibration_source": str(calibration_source) if calibrated else None,
            "compass_direction_available": False,
        },
        "track_count": len(public_tracks),
        "frame_object_counts": frame_counts,
        "tracks": public_tracks,
        "split_event_count": len(split_events),
        "merge_event_count": len(merge_events),
        "split_events": split_events[:20],
        "merge_events": merge_events[:20],
        "association_method": (
            "real connected-mask IoU plus centroid distance, one-to-one Hungarian assignment, "
            f"maximum gap {maximum_gap_frames} frame(s)"
        ),
    }


def _track_similarity(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float, float]:
    start = max(first["start_frame"], second["start_frame"])
    end = min(first["end_frame"], second["end_frame"])
    overlap = max(0, end - start + 1)
    union = max(first["end_frame"], second["end_frame"]) - min(
        first["start_frame"], second["start_frame"]
    ) + 1
    temporal_overlap = overlap / union
    distance = float(
        np.linalg.norm(np.asarray(first["mean_centroid"]) - np.asarray(second["mean_centroid"]))
    )
    cost = (1.0 - temporal_overlap) + distance / 64.0
    return cost, temporal_overlap, distance


class UncertaintyAnalyzer(WorldComponent):
    """Analyze member spread, connected-object tracks, and cross-member consensus."""

    def analyze(
        self, event_id: str, threshold: float = 133.0, minimum_region_pixels: int = 1
    ) -> dict[str, Any]:
        """Return the existing pixel-wise and storm-object uncertainty summary."""
        arguments = {
            "event_id": event_id,
            "threshold": threshold,
            "minimum_region_pixels": minimum_region_pixels,
        }
        return self._call(
            "uncertainty.analyze", arguments, forecast.analyze_ensemble_uncertainty
        )

    def uploaded_sequence(
        self,
        upload_id: str,
        fields: list[str],
        lead_minutes: list[int],
        threshold: float = 133.0,
        minimum_region_pixels: int = 1,
        ensemble_members: int = 10,
    ) -> dict[str, Any]:
        """Fetch agent-selected per-member values for an uploaded CasCast run.

        ``fields`` is required and may contain mean_vil, max_vil,
        exceeding_pixel_count, exceeding_area_ratio, threshold_centroid_x,
        threshold_centroid_y, largest_region_pixel_count,
        largest_region_area_ratio, largest_region_centroid_x, or
        largest_region_centroid_y. ``lead_minutes`` is also required and may
        contain any 5-minute forecast leads from 5 through 60.

        This tool deliberately returns records only. The generated run() code
        must choose and calculate any range, median, standard deviation,
        relative spread, trend, or member-to-member distance needed to answer
        the question. Use this instead of analyze(event_id=...) for uploads.
        """
        arguments = {
            "upload_id": upload_id,
            "fields": fields,
            "lead_minutes": lead_minutes,
            "threshold": threshold,
            "minimum_region_pixels": minimum_region_pixels,
            "ensemble_members": ensemble_members,
        }
        return self._call(
            "uncertainty.uploaded_sequence",
            arguments,
            forecast.query_uploaded_ensemble_member_values,
        )

    def uploaded_raw(
        self,
        upload_id: str,
        lead_minutes: list[int],
        ensemble_members: int = 10,
        horizon_hours: float = 1.0,
        include_ground_truth: bool = False,
        force_live: bool = False,
    ) -> dict[str, Any]:
        """Return raw CasCast member grids for selected forecast leads.

        This is the preferred primitive for open-ended ensemble questions.
        ``horizon_hours`` must be 1.0, 2.0, or 3.0. For a two- or three-hour
        rolling request, leads may extend through 120 or 180 minutes. The same
        uploaded 13-frame history conditions consecutive 60-minute blocks.
        It accepts no fields or threshold and computes no statistics. The
        generated ``run()`` code must choose fields, comparisons, and
        statistics from the raw member pixels. Up to twelve leads are accepted
        in one call; the trusted wrapper chunks backend reads while returning
        one aligned result, so generated code must not make tool calls in a
        loop. The exact response is
        ``records``, one item per member; each record has ``lead_minutes`` and
        a positionally aligned read-only float32 ``frames`` array shaped
        ``[selected_lead, row, column]`` inside
        the isolated worker. Analysis values retain CasCast decoded precision;
        uint8 conversion is for display only. Every member includes its seed,
        output_type, and is_deterministic=false. There are no dynamic keys such
        as ``lead_60``.
        ``records`` contains one item per member, not one item per lead. Each
        record's ``lead_minutes`` and the first ``frames`` axis are positionally aligned;
        use the lead's list index to select its frame. Do not treat ``frames``
        as a dictionary keyed by lead.
        ``records[i]['frames']`` is an ndarray: never write ``if frames`` or
        ``if not frames``. Check ``len(records)``, the documented shape, or use
        ``arrays.any/all/count_nonzero``. Array transport is automatic; do not
        pass an ``array_transport`` argument. For all 12 one-hour leads, call
        this method once with ``lead_minutes=[5,10,...,60]``.
        By default this uncertainty primitive does not load or return held-out
        future observations. Set ``include_ground_truth=true`` only for an
        explicit verification request; normally prefer
        ``metrics.uploaded_raw_pair`` so prediction/truth alignment and the
        missing-value mask are explicit.
        """
        arguments = {
            "upload_id": upload_id,
            "lead_minutes": lead_minutes,
            "ensemble_members": ensemble_members,
            "horizon_hours": horizon_hours,
            "include_ground_truth": include_ground_truth,
            "force_live": bool(force_live),
            # Internal transport choice: keep the cube as an ndarray inside the
            # isolated worker instead of expanding millions of JSON scalars.
            "array_transport": True,
        }
        return self._call(
            "uncertainty.uploaded_raw",
            arguments,
            _uploaded_raw_contract,
        )

    def track_objects(
        self,
        event_id: str,
        source: str = "ensemble_mean",
        member: int = 1,
        threshold: float = 133.0,
        minimum_region_pixels: int = 1,
        pixel_size_km: float | None = None,
        calibration_source: str = "",
    ) -> dict[str, Any]:
        """Track per-frame objects in ensemble mean, ground truth, or one member.

        Member numbering is one-based. Results stay in pixels unless an
        explicit positive pixel size and its calibration provenance are passed.
        """
        arrays, _ = ensemble._load_cascast_ensemble(event_id)
        if source == "ensemble_mean":
            fields = arrays["ensemble_mean"]
        elif source == "ground_truth":
            fields = arrays["ground_truth"]
        elif source == "member":
            ensemble_count = int(arrays["ensemble_members"].shape[0])
            if not 1 <= int(member) <= ensemble_count:
                raise ValueError(f"member must be between 1 and {ensemble_count}")
            fields = arrays["ensemble_members"][int(member) - 1]
        else:
            raise ValueError("source must be ensemble_mean, ground_truth, or member")
        result = {
            "status": "ok",
            "task": "storm_object_tracking",
            "event_id": event_id,
            "source": source,
            "member": int(member) if source == "member" else None,
            "threshold": threshold,
            **track_fields(
                fields,
                threshold,
                minimum_region_pixels,
                pixel_size_km=pixel_size_km,
                calibration_source=calibration_source,
            ),
        }
        self._recorder(
            "uncertainty.track_objects",
            {
                "event_id": event_id,
                "source": source,
                "member": member,
                "threshold": threshold,
                "minimum_region_pixels": minimum_region_pixels,
                "pixel_size_km": pixel_size_km,
                "calibration_source": calibration_source,
            },
            result,
        )
        return result

    def match_ensemble_objects(
        self,
        event_id: str,
        threshold: float = 133.0,
        minimum_region_pixels: int = 1,
    ) -> dict[str, Any]:
        """Match storm tracks across all available members into consensus objects.

        Members are processed one at a time and matched one-to-one to existing
        consensus prototypes using temporal overlap and mean-path distance.
        Member support is an ensemble fraction, not a calibrated hazard probability.
        """
        arrays, _ = ensemble._load_cascast_ensemble(event_id)
        ensemble_count = int(arrays["ensemble_members"].shape[0])
        member_tracks = [
            track_fields(member, threshold, minimum_region_pixels)["tracks"]
            for member in arrays["ensemble_members"]
        ]
        clusters: list[dict[str, Any]] = []
        for member_index, tracks in enumerate(member_tracks, start=1):
            if not clusters:
                clusters = [{"members": [member_index], "tracks": [track]} for track in tracks]
                continue
            prototypes = []
            for cluster in clusters:
                prototypes.append(
                    {
                        "start_frame": int(round(np.mean([t["start_frame"] for t in cluster["tracks"]]))),
                        "end_frame": int(round(np.mean([t["end_frame"] for t in cluster["tracks"]]))),
                        "mean_centroid": np.mean(
                            [t["mean_centroid"] for t in cluster["tracks"]], axis=0
                        ).tolist(),
                    }
                )
            cost = np.full((len(clusters), len(tracks)), 1e6, dtype=np.float64)
            for row, prototype in enumerate(prototypes):
                for column, track in enumerate(tracks):
                    value, temporal_overlap, distance = _track_similarity(prototype, track)
                    if temporal_overlap >= 0.20 and distance <= 64.0:
                        cost[row, column] = value
            matched_tracks: set[int] = set()
            for row, column in _assign(cost):
                if cost[row, column] >= 1e5:
                    continue
                clusters[row]["members"].append(member_index)
                clusters[row]["tracks"].append(tracks[column])
                matched_tracks.add(column)
            for column, track in enumerate(tracks):
                if column not in matched_tracks:
                    clusters.append({"members": [member_index], "tracks": [track]})

        consensus = []
        for index, cluster in enumerate(clusters, start=1):
            tracks = cluster["tracks"]
            centroids = np.asarray([track["mean_centroid"] for track in tracks], dtype=np.float64)
            areas = np.asarray([track["maximum_pixel_count"] for track in tracks], dtype=np.float64)
            durations = np.asarray([track["duration_minutes"] for track in tracks], dtype=np.float64)
            consensus.append(
                {
                    "consensus_id": f"C{index}",
                    "member_count": len(cluster["members"]),
                    "member_support_fraction": round(
                        len(cluster["members"]) / float(ensemble_count), 3
                    ),
                    "members": cluster["members"],
                    "mean_centroid": [round(value, 3) for value in centroids.mean(axis=0)],
                    "centroid_spread_pixels": round(
                        float(np.sqrt(np.var(centroids, axis=0).sum())), 3
                    ),
                    "maximum_area_pixels_q10_q50_q90": [
                        round(float(np.quantile(areas, quantile)), 3)
                        for quantile in (0.10, 0.50, 0.90)
                    ],
                    "duration_minutes_q10_q50_q90": [
                        round(float(np.quantile(durations, quantile)), 3)
                        for quantile in (0.10, 0.50, 0.90)
                    ],
                }
            )
        consensus.sort(
            key=lambda item: (item["member_count"], item["maximum_area_pixels_q10_q50_q90"][1]),
            reverse=True,
        )
        for index, item in enumerate(consensus, start=1):
            item["consensus_id"] = f"C{index}"
        result = {
            "status": "ok",
            "task": "cross_ensemble_object_matching",
            "capabilities": ["grid_object_tracking", "ensemble_object_agreement"],
            "provenance": {
                "prediction_source": "saved CasCast ensemble members",
                "spatial_units": "pixels",
                "physical_calibration_available": False,
                "compass_direction_available": False,
                "member_support_is_calibrated_probability": False,
            },
            "event_id": event_id,
            "threshold": threshold,
            "ensemble_count": ensemble_count,
            "member_track_counts": [len(tracks) for tracks in member_tracks],
            "consensus_object_count": len(consensus),
            "consensus_objects": consensus[:20],
            "matching_method": (
                "member-wise one-to-one Hungarian assignment using temporal overlap and "
                "mean-track centroid distance"
            ),
            "interpretation_limit": "Member support is ensemble agreement, not observed probability.",
        }
        self._recorder(
            "uncertainty.match_ensemble_objects",
            {
                "event_id": event_id,
                "threshold": threshold,
                "minimum_region_pixels": minimum_region_pixels,
            },
            result,
        )
        return result
