from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


# Ten deterministic samples spread across the available HDF5 rows.  Keeping
# the row choices explicit makes the example set reproducible while still
# deriving every image from the user's HDF5 file.
DEFAULT_ROWS = (0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2323)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export split history/ground-truth VIL examples from SEVIR HDF5.")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    with h5py.File(args.hdf5, "r") as handle:
        ids = [value.decode("ascii") for value in handle["id"][:]]
        for sample_number, row in enumerate(DEFAULT_ROWS, start=1):
            if row < 0 or row >= len(ids):
                raise ValueError(f"HDF5 row {row} is not present in {args.hdf5.name}")
            event_id = ids[row]
            strength_label = f"sample_{sample_number:02d}"
            # The HDF5 event contains 49 frames at -120..+120 minutes.
            # Use -60..0 as the model input and +5..+60 as held-out truth.
            history = np.moveaxis(np.asarray(handle["vil"][row, :, :, 12:25], dtype=np.uint8), -1, 0)
            ground_truth = np.moveaxis(np.asarray(handle["vil"][row, :, :, 25:37], dtype=np.uint8), -1, 0)
            event_dir = args.output_dir / event_id
            history_dir = event_dir / "history"
            truth_dir = event_dir / "ground_truth"
            history_dir.mkdir(parents=True, exist_ok=True)
            truth_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, frame in enumerate(history):
                relative_minutes = -60 + frame_index * 5
                filename = f"frame_{frame_index + 1:02d}_{relative_minutes:+04d}min.png"
                Image.fromarray(frame, mode="L").save(history_dir / filename)
            for frame_index, frame in enumerate(ground_truth):
                relative_minutes = 5 + frame_index * 5
                filename = f"frame_{frame_index + 1:02d}_{relative_minutes:+04d}min.png"
                Image.fromarray(frame, mode="L").save(truth_dir / filename)
            # A single-frame upload example is the final observed frame (0 min)
            # from the same event, so it is guaranteed to be traceable to the
            # sequence and to the source HDF5 row.
            single_dir = args.output_dir / "single_frames"
            single_dir.mkdir(parents=True, exist_ok=True)
            single_filename = f"{event_id}_frame_13_+000min.png"
            Image.fromarray(history[-1], mode="L").save(single_dir / single_filename)
            all_frames = np.concatenate([history, ground_truth], axis=0)
            manifest.append(
                {
                    "event_id": event_id,
                    "strength_label": strength_label,
                    "hdf5_row": row,
                    "folder": event_dir.name,
                    "history_folder": "history",
                    "ground_truth_folder": "ground_truth",
                    "history_frame_count": 13,
                    "ground_truth_frame_count": 12,
                    "history_minutes": list(range(-60, 5, 5)),
                    "ground_truth_minutes": list(range(5, 65, 5)),
                    "shape": [384, 384],
                    "min_vil": int(all_frames.min()),
                    "max_vil": int(all_frames.max()),
                    "mean_vil": round(float(all_frames.mean()), 4),
                    "vil_ge_133_area_ratio": round(float((all_frames >= 133).mean()), 6),
                    "split": "history is input; ground_truth is held out future for evaluation",
                    "single_frame": f"single_frames/{single_filename}",
                }
            )
    (args.output_dir / "hdf5_sequence_manifest.json").write_text(
        json.dumps({"source_hdf5": str(args.hdf5.resolve()), "sequences": manifest}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"exported": len(manifest), "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
