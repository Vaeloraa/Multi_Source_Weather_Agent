from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "cascast_evaluation_report.json"
OUTPUT = ROOT / "data" / "evaluation_exports"


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = OUTPUT / name
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not REPORT.exists():
        raise SystemExit("请先生成data/cascast_evaluation_report.json")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    threshold_rows = []
    for model, by_threshold in report["aggregate_metrics"].items():
        for threshold, metrics in by_threshold.items():
            threshold_rows.append({"model": model, "threshold": threshold, **metrics})
    write_csv("threshold_metrics.csv", threshold_rows)

    lead_rows = []
    for model, by_threshold in report["lead_time_metrics"].items():
        for threshold, records in by_threshold.items():
            for record in records:
                lead_rows.append({"model": model, "threshold": threshold, **record})
    write_csv("lead_time_metrics.csv", lead_rows)

    event_rows = []
    for event in report["events"]:
        row: dict[str, object] = {
            "event_id": event["event_id"],
            "sample_index": event["sample_index"],
            "event_type": event["event_type"],
        }
        for model in ("cascast", "persistence"):
            metrics = event[model]
            for key in ("csi_at_threshold", "pod_at_threshold", "far_at_threshold", "mse_raw_vil", "rmse_raw_vil", "mae_raw_vil"):
                row[f"{model}_{key}"] = metrics[key]
        row["predicted_track_count"] = event["predicted_storm_tracks"]["track_count"]
        row["observed_track_count"] = event["observed_storm_tracks"]["track_count"]
        row["object_csi"] = event["storm_object_verification"]["object_csi"]
        row["object_centroid_error_pixels"] = event["storm_object_verification"]["mean_matched_centroid_error_pixels"]
        event_rows.append(row)
    write_csv("event_metrics.csv", event_rows)

    write_csv(
        "ensemble_member_metrics.csv",
        report["ensemble_member_performance"]["members"],
    )

    reliability_rows = []
    for threshold, records in report["ensemble_validation"]["reliability"].items():
        for record in records:
            reliability_rows.append({"threshold": threshold, **record})
    write_csv("ensemble_reliability.csv", reliability_rows)

    primary = str(int(float(report["primary_threshold"])))
    cascast = report["aggregate_metrics"]["cascast_ensemble_mean"][primary]
    baseline = report["aggregate_metrics"]["persistence"][primary]
    validation = report["ensemble_validation"]
    objects = report["storm_object_evaluation"]
    summary = f"""# CasCast × SEVIR 10事件评价摘要

- 数据：10个SEVIR VIL测试事件，每个事件13帧历史、10路CasCast ensemble、12帧预测和12帧ground truth。
- 主阈值：VIL ≥ {primary}。
- CasCast ensemble mean：CSI {cascast['csi']}，POD {cascast['pod']}，FAR {cascast['far']}，RMSE {cascast['rmse_raw_vil']}，MAE {cascast['mae_raw_vil']}。
- Persistence：CSI {baseline['csi']}，POD {baseline['pod']}，FAR {baseline['far']}，RMSE {baseline['rmse_raw_vil']}，MAE {baseline['mae_raw_vil']}。
- Ensemble验证：pixel std–absolute error Pearson {validation['pixel_std_absolute_error_pearson']}，80%中心区间实际覆盖率 {validation['central_80_interval_coverage']}，spread-skill ratio {validation['spread_skill_ratio']}。
- Storm object：CSI {objects['object_csi']}，POD {objects['object_pod']}，FAR {objects['object_far']}，匹配对象中心平均误差 {objects['mean_matched_centroid_error_pixels']}像素。

## 解释

CasCast在这10个事件上明显优于Persistence。标准差与真实误差呈正相关，说明ensemble分歧包含有用误差信息；但80%区间仅覆盖约{validation['central_80_interval_coverage'] * 100:.1f}%的有效像素，且spread-skill ratio明显小于1，表明ensemble偏窄、离散度不足。以上结果只代表当前10个事件，不是完整SEVIR测试集论文复现。
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    print(f"Exported evaluation tables to {OUTPUT}")


if __name__ == "__main__":
    main()
