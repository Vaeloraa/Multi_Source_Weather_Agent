"""Read-only readiness check for the R190627 end-to-end demonstration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "data" / "demo_cases" / "r19062708588472.json"


def _npz_shapes(path: Path, required: dict[str, tuple[int, ...]]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "文件缺失"
    try:
        with np.load(path, allow_pickle=False) as package:
            wrong = [
                f"{name}={tuple(package[name].shape) if name in package else '缺失'}"
                for name, shape in required.items()
                if name not in package or tuple(package[name].shape) != shape
            ]
    except Exception as error:
        return False, f"无法读取：{type(error).__name__}"
    return (not wrong), ("结构正确" if not wrong else "；".join(wrong))


def _aligned_aurora(cache: dict[str, str]) -> tuple[bool, str]:
    summary_path = ROOT / cache["aurora_aligned_summary"]
    package_path = ROOT / cache["aurora_aligned"]
    if not summary_path.is_file() or not package_path.is_file():
        return False, "03:00Z 时间对齐预测包或摘要缺失"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        init_time = datetime.fromisoformat(
            str(summary["event_input_time"]).replace("Z", "+00:00")
        )
        if init_time.tzinfo is None:
            init_time = init_time.replace(tzinfo=timezone.utc)
        lead6 = datetime.fromisoformat(
            str(summary["target_point_forecasts"]["6"]["time_utc"]).replace("Z", "+00:00")
        )
        if lead6.tzinfo is None:
            lead6 = lead6.replace(tzinfo=timezone.utc)
        event = datetime.fromisoformat(
            str(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["selected_event_anchor_time_utc"])
            .replace("Z", "+00:00")
        )
        gap_minutes = abs((lead6 - event).total_seconds()) / 60.0
        leads = [int(value) for value in summary.get("completed_leads_hours", [])]
        valid = (
            init_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M") == "2019-06-27T03:00"
            and lead6.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M") == "2019-06-27T09:00"
            and leads == [6, 12, 18, 24]
            and gap_minutes <= 5.0
        )
        return valid, (
            f"03:00Z初始化，+6h=09:00Z，与事件相差{gap_minutes:.2f}分钟，"
            f"时效{leads}"
        )
    except Exception as error:
        return False, f"无法核验时间契约：{type(error).__name__}: {error}"


def inspect() -> list[tuple[str, bool, str]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cache = config["cache_contract"]
    foundation_root = ROOT / cache["foundation_root"]
    manifests = list(foundation_root.glob("*/live_manifest.json"))
    aligned_ok, aligned_note = _aligned_aurora(cache)
    verification = list(foundation_root.glob("*/verification/aurora_vs_era5_verification.json"))
    primary_ok, primary_note = _npz_shapes(
        ROOT / cache["cascast_primary"],
        {"history": (13, 384, 384), "ensemble_members": (10, 12, 384, 384), "ground_truth": (12, 384, 384)},
    )
    boundary_ok, boundary_note = _npz_shapes(
        ROOT / cache["cascast_boundary_demo"],
        {"history": (13, 384, 384), "ensemble_members": (10, 36, 384, 384)},
    )
    event_ok, event_note = _npz_shapes(
        ROOT / cache["sevir_event"], {"frames": (384, 384, 49)},
    )
    history_count = len(list((ROOT / cache["history_images"]).glob("*.png")))
    truth_count = len(list((ROOT / cache["verification_images"]).glob("*.png")))
    return [
        ("案例配置", CONFIG_PATH.is_file(), "R19062708588472"),
        ("ERA5 区域输入缓存", bool(manifests), f"清单 {len(manifests)}"),
        ("Aurora 时间对齐区域预测", aligned_ok, aligned_note),
        ("Aurora 后续 ERA5 验证", bool(verification), f"验证文件 {len(verification)}"),
        ("CasCast 未来 60 分钟", primary_ok, primary_note),
        ("CasCast 3 小时边界包", boundary_ok, boundary_note + "；只作时效边界，不作可靠主结论"),
        ("SEVIR 49 帧事件实况", event_ok, event_note),
        ("此前一小时展示帧", history_count == 13, f"{history_count}/13 帧"),
        ("后一小时验证帧", truth_count == 12, f"{truth_count}/12 帧"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 R190627 Demo 缓存是否齐全")
    parser.add_argument("--require-ready", action="store_true", help="任一核心阶段缺失时返回非零")
    args = parser.parse_args()
    rows = inspect()
    print("R190627 区域发现—局地追踪 Demo 预检")
    for name, passed, note in rows:
        print(f"[{'就绪' if passed else '缺失'}] {name}：{note}")
    ready = all(passed for _, passed, _ in rows)
    print("结论：" + ("缓存链路完整，可走快速 Demo。" if ready else "缓存链路不完整，缺失阶段必须明确显示。"))
    return 2 if args.require_ready and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
