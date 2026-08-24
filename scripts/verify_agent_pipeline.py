from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sevir_world.code_execution.client import ExecutionClient  # noqa: E402


PROGRAM = '''def run(ds, uncertainty, visualizer):
    raw = uncertainty.uploaded_raw(upload_id=ds[0]["upload_id"], lead_minutes=[60], ensemble_members=10)
    frames = [record["frames"][0] for record in raw["records"]]
    member_count = len(frames)
    height = len(frames[0])
    width = len(frames[0][0])
    ensemble_mean = []
    spread = []
    for row_index in range(height):
        mean_row = []
        spread_row = []
        for column_index in range(width):
            values = [frame[row_index][column_index] for frame in frames]
            pixel_mean = sum(values) / member_count
            variance = sum([(value - pixel_mean) ** 2 for value in values]) / member_count
            mean_row.append(pixel_mean)
            spread_row.append(variance ** 0.5)
        ensemble_mean.append(mean_row)
        spread.append(spread_row)
    truth_available = raw["ground_truth_available"]
    valid_count = 0
    error_sum = 0.0
    member_error_sums = [0.0 for frame in frames]
    absolute_error = []
    panels = [
        {"label": "Ensemble mean +60", "values": ensemble_mean, "cmap": "turbo", "vmin": 0, "vmax": 254, "colorbar_label": "encoded VIL"},
        {"label": "Member spread +60", "values": spread, "cmap": "magma", "vmin": 0, "vmax": 60, "colorbar_label": "member std"}
    ]
    if truth_available:
        truth = raw["ground_truth"]["frames"][0]
        for row_index in range(height):
            error_row = []
            for column_index in range(width):
                truth_value = truth[row_index][column_index]
                error = 0.0
                if truth_value != 255:
                    error = abs(ensemble_mean[row_index][column_index] - truth_value)
                    error_sum += error
                    valid_count += 1
                    for member_index in range(member_count):
                        member_error_sums[member_index] += abs(frames[member_index][row_index][column_index] - truth_value)
                error_row.append(error)
            absolute_error.append(error_row)
        panels.extend([
            {"label": "Ground Truth +60", "values": truth, "cmap": "turbo", "vmin": 0, "vmax": 254, "colorbar_label": "encoded VIL"},
            {"label": "Absolute error +60", "values": absolute_error, "cmap": "magma", "vmin": 0, "vmax": 100, "colorbar_label": "absolute error"}
        ])
    chart = visualizer.grid_panels(panels=panels, title="CasCast ensemble and exact HDF5 truth")
    return {
        "ground_truth_available": truth_available,
        "member_count": member_count,
        "mean_pixel_spread": sum([sum(row) for row in spread]) / (height * width),
        "ensemble_mean_mae": error_sum / valid_count if valid_count else None,
        "member_mae": [value / valid_count for value in member_error_sums] if valid_count else [],
        "artifact": chart,
        "provenance": raw["provenance"]
    }
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_agent_pipeline.py UPLOAD_ID")
    client = ExecutionClient(timeout_seconds=30, live_timeout_seconds=30)
    result = client.execute(
        PROGRAM,
        {"weather_model": "cascast_live"},
        data_descs=[{"source": "upload", "upload_id": sys.argv[1]}],
        selected_tools=["ds", "uncertainty", "visualizer"],
    )
    summary = {
        "status": result.get("status"),
        "error": result.get("error"),
        "elapsed_ms": result.get("elapsed_ms"),
        "result": result.get("result"),
        "artifacts": result.get("generated_artifacts", []),
        "calls": result.get("calls", []),
    }
    print(json.dumps(summary, ensure_ascii=False))
    client.close()
    if result.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
