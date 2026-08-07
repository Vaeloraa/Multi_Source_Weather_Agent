from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

import sevir_tools

from .base import WorldComponent

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only in minimal deployments
    linear_sum_assignment = None


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
    minimum_region_pixels: int = 16,
    maximum_gap_frames: int = 1,
    pixel_size_km: float = 1.0,
) -> dict[str, Any]:
    """Track real connected masks with one-to-one assignment and one-frame gaps."""
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
        segment_distance = (
            np.linalg.norm(np.diff(centroids, axis=0), axis=1) * pixel_size_km
            if len(centroids) > 1
            else np.asarray([], dtype=np.float64)
        )
        elapsed_hours = max((frames[-1] - frames[0]) * 5 / 60.0, 0.0)
        path_km = float(segment_distance.sum())
        observations = [
            {
                "frame_index": frame_index,
                "lead_minutes": (frame_index + 1) * 5,
                "centroid": [round(value, 3) for value in item["centroid"]],
                "pixel_count": item["pixel_count"],
                "area_km2_approx": round(item["pixel_count"] * pixel_size_km**2, 3),
                "maximum_vil": round(item["maximum_vil"], 3),
            }
            for frame_index, item in zip(frames, objects)
        ]
        public_tracks.append(
            {
                "track_id": f"T{index}",
                "start_frame": frames[0],
                "end_frame": frames[-1],
                "duration_minutes": len(frames) * 5,
                "maximum_gap_frames": max(np.diff(frames), default=1) - 1,
                "maximum_pixel_count": max(item["pixel_count"] for item in objects),
                "path_length_km_approx": round(path_km, 3),
                "mean_speed_kmh_approx": round(path_km / elapsed_hours, 3) if elapsed_hours else 0.0,
                "mean_centroid": [round(value, 3) for value in centroids.mean(axis=0)],
                "observations": observations,
            }
        )
    public_tracks.sort(
        key=lambda item: (item["duration_minutes"], item["maximum_pixel_count"]), reverse=True
    )
    return {
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
        self, event_id: str, threshold: float = 133.0, minimum_region_pixels: int = 16
    ) -> dict[str, Any]:
        """Return the existing pixel-wise and storm-object uncertainty summary."""
        arguments = {
            "event_id": event_id,
            "threshold": threshold,
            "minimum_region_pixels": minimum_region_pixels,
        }
        return self._call(
            "uncertainty.analyze", arguments, sevir_tools.analyze_ensemble_uncertainty
        )

    def track_objects(
        self,
        event_id: str,
        source: str = "ensemble_mean",
        member: int = 1,
        threshold: float = 133.0,
        minimum_region_pixels: int = 16,
    ) -> dict[str, Any]:
        """Track per-frame objects in ensemble mean, ground truth, or one member.

        Member numbering is one-based. Distances assume the native VIL grid is
        approximately 1 km per pixel.
        """
        arrays, _ = sevir_tools._load_cascast_ensemble(event_id)
        if source == "ensemble_mean":
            fields = arrays["ensemble_mean"]
        elif source == "ground_truth":
            fields = arrays["ground_truth"]
        elif source == "member":
            if not 1 <= int(member) <= arrays["ensemble_members"].shape[0]:
                raise ValueError("member must be between 1 and 10")
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
            **track_fields(fields, threshold, minimum_region_pixels),
        }
        self._recorder(
            "uncertainty.track_objects",
            {
                "event_id": event_id,
                "source": source,
                "member": member,
                "threshold": threshold,
                "minimum_region_pixels": minimum_region_pixels,
            },
            result,
        )
        return result

    def match_ensemble_objects(
        self,
        event_id: str,
        threshold: float = 133.0,
        minimum_region_pixels: int = 16,
    ) -> dict[str, Any]:
        """Match storm tracks across ten members into consensus objects.

        Members are processed one at a time and matched one-to-one to existing
        consensus prototypes using temporal overlap and mean-path distance.
        Member support is an ensemble fraction, not a calibrated hazard probability.
        """
        arrays, _ = sevir_tools._load_cascast_ensemble(event_id)
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
                    "member_support_fraction": round(len(cluster["members"]) / 10.0, 3),
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
            "event_id": event_id,
            "threshold": threshold,
            "ensemble_count": 10,
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
