from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from sevir_world.backends.forecast import get_uploaded_vil_sequence_forecast
from sevir_world.data_access.config import CASCAST_VENV_SITE_PACKAGES
from sevir_world.data_access.live_inference import _worker_environment
from sevir_world.tools.forecast import CasCastForecaster


class UploadedForecastRecoveryTests(unittest.TestCase):
    def test_worker_environment_exposes_cascast_venv_packages(self) -> None:
        environment = _worker_environment()
        python_paths = environment["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(python_paths[0], str(CASCAST_VENV_SITE_PACKAGES))

    def test_tool_normalizes_cascast_alias_for_uploaded_sequence(self) -> None:
        recorder_calls = []
        forecaster = CasCastForecaster(lambda *args, **kwargs: recorder_calls.append(args))
        with patch(
            "sevir_world.tools.forecast.forecast.get_uploaded_vil_sequence_forecast",
            return_value={"status": "ok"},
        ) as backend:
            result = forecaster.uploaded_sequence("a" * 32, model_name="cascast")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(backend.call_args.kwargs["model_name"], "cascast_live")

    def test_backend_normalizes_cascast_alias_for_uploaded_sequence(self) -> None:
        sequence = np.zeros((13, 2, 2), dtype=np.uint8)
        arrays = {
            "ensemble_mean": np.zeros((12, 2, 2), dtype=np.float32),
            "ensemble_members": np.zeros((1, 12, 2, 2), dtype=np.float32),
            "member_seed": np.asarray([7], dtype=np.int64),
        }
        upload_metadata = {
            "history_minutes": list(range(-60, 5, 5)),
            "source": {"kind": "test_upload"},
        }
        live_metadata = {"cache_file": "test-cache.npz", "ensemble_count": 1}

        with (
            patch(
                "sevir_world.backends.forecast._load_uploaded_vil_sequence",
                return_value=(sequence, upload_metadata),
            ),
            patch(
                "sevir_world.backends.forecast.run_uploaded_cascast_inference",
                return_value={"cache_hit": True},
            ) as inference,
            patch(
                "sevir_world.backends.forecast.load_uploaded_cascast_ensemble",
                return_value=(arrays, live_metadata),
            ),
        ):
            result = get_uploaded_vil_sequence_forecast(
                "b" * 32,
                model_name="cascast",
                ensemble_members=1,
                array_transport=True,
            )

        self.assertEqual(result["model_name"], "cascast_live")
        self.assertEqual(result["forecast_frame_count"], 12)
        inference.assert_called_once_with(
            "b" * 32, ensemble_members=1, rolling_segments=1
        )


if __name__ == "__main__":
    unittest.main()
