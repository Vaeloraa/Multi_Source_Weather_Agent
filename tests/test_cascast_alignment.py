from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sevir_world.data_access import uploads
from sevir_world.data_access.ensemble import _ensemble_event
from sevir_world.data_access.uploads import load_uploaded_sequence_hdf5_ground_truth
from sevir_world.domain.cascast_alignment import (
    build_validation_manifest,
    parse_sample_identity,
)
from sevir_world.tools.metrics import VerificationMetrics


class CasCastAlignmentTests(unittest.TestCase):
    def test_prepared_r190_cache_is_registered_for_ordinary_tools(self) -> None:
        event = _ensemble_event("R19062708588472")
        self.assertEqual(event["event_id"], "R19062708588472")
        self.assertEqual(event["cache_root"], "live_cascast_cache")
        self.assertEqual(event["history_frames"], 13)
        self.assertEqual(event["forecast_frames"], 12)
        self.assertEqual(event["ensemble_count"], 10)

    def test_sample_identity_prefers_metadata_and_parses_name_fallbacks(self) -> None:
        parsed = parse_sample_identity(
            "R19062708588472_segment-1_row-2323.npz",
            {"h5_file": "vil.h5", "segment": 2, "file_index": 7},
        )
        self.assertEqual(parsed, {
            "event_id": "R19062708588472",
            "h5_file": "vil.h5",
            "file_index": 7,
            "segment": 2,
        })

    def test_validation_limits_follow_absolute_segment_frames(self) -> None:
        expected = {
            0: (list(range(0, 13)), 180),
            1: (list(range(12, 25)), 120),
            2: (list(range(24, 37)), 60),
        }
        for segment, (input_frames, maximum) in expected.items():
            with self.subTest(segment=segment):
                manifest = build_validation_manifest(
                    {
                        "event_id": "R19062708588472",
                        "h5_file": "vil.h5",
                        "file_index": 2323,
                        "segment": segment,
                    },
                    forecast_mode="free_running",
                    requested_lead_minutes=[60, 120, 180],
                )
                self.assertEqual(manifest["input_frames"], input_frames)
                self.assertEqual(manifest["max_valid_lead_minutes"], maximum)
                self.assertEqual(manifest["standard_max_valid_lead_minutes"], 60)
                self.assertEqual(
                    manifest["unverified_lead_minutes"],
                    [lead for lead in (60, 120, 180) if lead > maximum],
                )

    def test_ground_truth_loader_returns_partial_segment_one_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vil.h5"
            values = np.arange(49, dtype=np.uint8)[None, None, None, :]
            with uploads.h5py.File(path, "w") as handle:
                handle.create_dataset("vil", data=values)
            match = {
                "status": "matched",
                "event_id": "R19062708588472",
                "h5_file": "vil.h5",
                "file_index": 0,
                "hdf5_row": 0,
                "source_hdf5": str(path),
                "segment": 1,
                "method": "test_exact_pixels",
                "confidence": 1.0,
                "history_frame_indices": list(range(12, 25)),
            }
            with patch(
                "sevir_world.data_access.uploads.match_uploaded_sequence_hdf5",
                return_value=match,
            ):
                result = load_uploaded_sequence_hdf5_ground_truth(
                    "a" * 32, [5, 60, 120, 125, 180]
                )
        self.assertTrue(result["available"])
        self.assertEqual(result["segment"], 1)
        self.assertEqual(result["lead_minutes"], [5, 60, 120])
        self.assertEqual(result["hdf5_frame_indices"], [25, 36, 48])
        self.assertEqual(result["unverified_lead_minutes"], [125, 180])
        self.assertEqual(result["max_valid_lead_minutes"], 120)
        self.assertEqual([frame[0][0] for frame in result["frames"]], [25, 36, 48])

    def test_uploaded_evaluation_scores_only_verified_leads(self) -> None:
        prediction = np.ones((36, 2, 2), dtype=np.float32)
        arrays = {
            "history": np.zeros((13, 2, 2), dtype=np.uint8),
            "ensemble_mean": prediction,
            "ensemble_members": prediction[None],
            "member_seed": np.asarray([7]),
        }
        verified = list(range(5, 125, 5))
        unverified = list(range(125, 185, 5))
        truth = {
            "available": True,
            "event_id": "R19062708588472",
            "h5_file": "vil.h5",
            "file_index": 2323,
            "segment": 1,
            "lead_minutes": verified,
            "unverified_lead_minutes": unverified,
            "hdf5_frame_indices": list(range(25, 49)),
            "max_valid_lead_minutes": 120,
            "validation_manifest": {"segment": 1, "max_valid_lead_minutes": 120},
            "frames": np.zeros((24, 2, 2), dtype=np.uint8).tolist(),
        }
        metrics = VerificationMetrics(lambda *_args, **_kwargs: None)
        with (
            patch(
                "sevir_world.tools.metrics.run_uploaded_cascast_inference",
                return_value={"cache_hit": True},
            ),
            patch(
                "sevir_world.tools.metrics.load_uploaded_cascast_ensemble",
                return_value=(arrays, {"cache_file": "roll3.npz"}),
            ),
            patch(
                "sevir_world.tools.metrics.load_uploaded_sequence_hdf5_ground_truth",
                return_value=truth,
            ),
        ):
            result = metrics.evaluate_uploaded(
                "a" * 32, threshold=1.0, ensemble_members=1, horizon_hours=3.0
            )
        self.assertEqual(result["forecast_mode"], "free_running")
        self.assertEqual(result["verified_lead_minutes"], verified)
        self.assertEqual(result["unverified_lead_minutes"], unverified)
        self.assertEqual(len(result["per_lead"]), 24)
        self.assertEqual(result["aggregate"]["mae"], 1.0)
        self.assertEqual(result["aggregate"]["bias"], 1.0)
        self.assertFalse(result["ground_truth_complete_for_horizon"])


if __name__ == "__main__":
    unittest.main()
