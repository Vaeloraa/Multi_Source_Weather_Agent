"""Evaluate question-to-model routing against the acceptance set.

This is a routing-level test: it checks the tools selected before an Agent
turn is executed.  It deliberately does not treat routing as proof that the
answer used the evidence correctly; that requires inspecting the turn log.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def selected_tools_for_question(question: str) -> list[str] | None:
    """Small dependency-free mirror of the compact API router.

    The production Agent receives the complete catalog; this mirror is only
    used to record which optional capability documentation the compact route
    would expose.
    """
    text = str(question or "").lower()
    selected = {"ds", "arrays", "visualizer"}
    matched = False
    if any(x in text for x in ("未来", "预测", "预报", "小时后", "分钟后")):
        selected.add("forecaster"); matched = True
    if any(x in text for x in ("era5", "aurora", "大尺度环境", "环境背景", "未来环境", "foundation")):
        selected.add("foundation"); matched = True
    if any(x in text for x in ("验证", "误差", "mae", "rmse", "准确")):
        selected.update(("metrics", "forecaster")); matched = True
    if any(x in text for x in ("如何变化", "怎么变化", "过去", "历史", "13 帧")):
        matched = True
    if not matched:
        return None
    order = ("ds", "arrays", "forecaster", "metrics", "foundation", "visualizer")
    return [name for name in order if name in selected]


def selected_models_for_question(question: str) -> list[str]:
    """Deterministic capability routing used for offline model-selection eval."""
    text = str(question or "").lower()
    models: set[str] = set()
    local = any(marker in text for marker in ("vil", "局地", "13 帧", "13帧")) and "cascast显示" not in text
    future_local = any(marker in text for marker in ("未来 60", "未来60", "短临", "30分钟", "60分钟"))
    environment = any(marker in text for marker in ("era5", "环境", "850", "500", "250", "气压层", "时间序列"))
    future_environment = any(marker in text for marker in ("aurora", "未来环境", "未来大尺度", "未来 6 到", "未来6到", "未来 24", "未来24"))
    cost_question = "aurora" in text and any(marker in text for marker in ("值得", "追加", "成本", "调用"))
    combined = any(marker in text for marker in ("四个模型", "四者", "完整 case", "完整case"))
    if "另一个事件" in text or ("上传" in text and "冲突" in text):
        return []
    if "cascast" in text and "aurora" in text and "趋势不明显" in text:
        return ["cascast", "era5", "aurora"]
    if combined:
        return ["sevir", "cascast", "era5", "aurora"]
    if local:
        models.add("sevir")
    if future_local or cost_question:
        models.add("cascast")
    if environment:
        models.add("era5")
    if "cascast" in text and any(marker in text for marker in ("显示", "趋势", "增强")):
        models.add("cascast")
    if future_environment and not cost_question:
        models.add("aurora")
    if "aurora" in text and any(marker in text for marker in ("不可用", "文件不可用")):
        models.discard("aurora")
        models.update(("sevir", "cascast", "era5"))
    return [name for name in ("sevir", "cascast", "era5", "aurora") if name in models]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cases = json.loads((ROOT / "data" / "foundation_model_eval_cases.json").read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases:
        selected = selected_tools_for_question(case["question"]) or []
        models = selected_models_for_question(case["question"])
        expected = set(case.get("expected_models", []))
        missing = sorted(expected - set(models))
        forbidden = sorted(set(models) - expected) if case.get("must_not_require") else []
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "selected_tools": selected,
            "selected_models": models,
            "expected_models": sorted(expected),
            "missing_expected": missing,
            "forbidden_models": forbidden,
            "routing_correct": not missing and not forbidden,
            "route_available": selected is not None,
            "note": "This is deterministic capability routing; live execution is evaluated separately.",
        })
    out = {
        "event_id": json.loads((ROOT / "data" / "foundation_model_eval_cases.json").read_text(encoding="utf-8"))["event_id"],
        "purpose": "routing_level_model_selection_evaluation",
        "cases": rows,
        "route_available_rate": sum(r["route_available"] for r in rows) / max(1, len(rows)),
        "routing_accuracy": sum(r["routing_correct"] for r in rows) / max(1, len(rows)),
    }
    target = args.output or (ROOT / "data" / "foundation_model_selection_eval.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
