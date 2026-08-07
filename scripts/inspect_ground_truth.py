from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_H5 = (
    ROOT
    / "cascast_ensemble_uncertainty"
    / "cascast_ensemble_10samples_384.h5"
)
GROUND_TRUTH_H5 = (
    ROOT
    / "cascast_ensemble_ground_truth"
    / "cascast_ground_truth_10samples_384.h5"
)


def main() -> None:
    with h5py.File(PREDICTION_H5, "r") as prediction_file, h5py.File(
        GROUND_TRUTH_H5, "r"
    ) as truth_file:
        prediction_indices = prediction_file["sample_index"][:]
        truth_indices = truth_file["sample_index"][:]
        truth = truth_file["ground_truth"][:]
        report = {
            "prediction_keys": sorted(prediction_file.keys()),
            "ground_truth_keys": sorted(truth_file.keys()),
            "sample_indices_equal": bool(
                np.array_equal(prediction_indices, truth_indices)
            ),
            "sample_indices": truth_indices.astype(int).tolist(),
            "ground_truth_shape": list(truth.shape),
            "ground_truth_dtype": str(truth.dtype),
            "ground_truth_min": int(truth.min()),
            "ground_truth_max": int(truth.max()),
            "missing_255_count": int((truth == 255).sum()),
            "missing_255_ratio": float((truth == 255).mean()),
            "per_sample": [
                {
                    "sample_index": int(sample_index),
                    "min": int(sample.min()),
                    "max": int(sample.max()),
                    "missing_255_count": int((sample == 255).sum()),
                }
                for sample_index, sample in zip(truth_indices, truth)
            ],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
