from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    package_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "source_packages"
        / "cascast_uncertainty"
    )
    h5_path = package_dir / "cascast_ensemble_10samples_384.h5"

    with h5py.File(h5_path, "r") as handle:
        print("datasets:")
        for name in handle:
            dataset = handle[name]
            print(f"  {name}: shape={dataset.shape}, dtype={dataset.dtype}")

        print("sample_index:", handle["sample_index"][:].tolist())
        print("member_seed:", handle["member_seed"][:].tolist())

        valid = True
        for sample_position in range(handle["ensemble_members"].shape[0]):
            members = handle["ensemble_members"][sample_position]
            stored_mean = handle["ensemble_mean"][sample_position]
            calculated_mean = members.mean(axis=0)
            max_mean_error = float(np.max(np.abs(stored_mean - calculated_mean)))
            finite = bool(np.isfinite(members).all() and np.isfinite(stored_mean).all())
            print(
                f"sample={sample_position} "
                f"min={float(members.min()):.6f} "
                f"max={float(members.max()):.6f} "
                f"finite={finite} "
                f"max_mean_error={max_mean_error:.8f}"
            )
            valid = valid and finite and max_mean_error < 1e-4

    print("validated:", valid)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
