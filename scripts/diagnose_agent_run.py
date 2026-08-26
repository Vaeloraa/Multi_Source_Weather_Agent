"""Run one real unified-Agent request with readable diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sevir_world.api.model_client import (
    FIXED_MODEL_NAME,
    FIXED_PROVIDER_NAME,
    PROVIDERS,
    fixed_model_selection,
    post_chat_completion,
)
from sevir_world.api.unified_service import bind_request_assets, execute_unified_request
import sevir_world.api.unified_service as unified_service


def _brief(value: Any, limit: int = 260) -> str:
    text = str(value or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> None:
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", default="R19062708588472")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--request-budget",
        type=float,
        default=30.0,
        help="Diagnostic-only total Agent budget in seconds.",
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=30.0,
        help="Diagnostic-only timeout for one fixed-model call.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Diagnostic-only fixed-model output ceiling.",
    )
    parser.add_argument("--show-model-actions", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    request_budget = max(1.0, float(args.request_budget))
    unified_service.MAX_AGENT_REQUEST_SECONDS = request_budget
    os.environ["SEVIR_AGENT_REQUEST_TIMEOUT_SECONDS"] = str(request_budget)
    os.environ["MODEL_API_TIMEOUT_SECONDS"] = str(max(1.0, float(args.model_timeout)))
    os.environ["QWEN_MAX_TOKENS"] = str(max(64, int(args.max_tokens)))

    provider_name, model = fixed_model_selection(PROVIDERS)
    if provider_name != FIXED_PROVIDER_NAME or model != FIXED_MODEL_NAME:
        raise RuntimeError("diagnostic fixed-model selection is inconsistent")
    payload: dict[str, Any] = {
        "event_id": args.event_id,
        "data_source": "cached",
        "data_policy": "cached",
        "weather_model": "cascast_live",
        "session_id": f"diagnostic-{int(time.time())}",
    }
    descriptions, tools = bind_request_assets(args.question, payload)
    print("数据资产：" + ", ".join(
        f"{item.get('source')}:{item.get('event_id') or item.get('upload_id') or '-'}"
        for item in descriptions
    ))
    print("注入工具：" + ", ".join(tools))
    print(f"固定模型：{provider_name}/{model}（失败不回退）")
    print(
        f"诊断预算：总计{request_budget:g}s；单次模型{float(args.model_timeout):g}s；"
        f"输出上限{int(args.max_tokens)} tokens"
    )
    print("开始统一 Agent…", flush=True)
    started = time.perf_counter()
    model_actions: list[dict[str, Any]] = []

    def complete(current_provider, current_model, messages, timeout_seconds):
        response = post_chat_completion(
            current_provider,
            current_model,
            messages,
            timeout_seconds=min(float(timeout_seconds), float(args.model_timeout)),
        )
        choice = (response.get("choices") or [{}])[0]
        content = str(choice.get("message", {}).get("content", ""))
        action = {
            "content": content,
            "finish_reason": choice.get("finish_reason"),
            "characters": len(content),
        }
        model_actions.append(action)
        if args.show_model_actions:
            print("\n--- 模型原始动作 ---")
            print(content)
            print(
                f"--- 结束；finish_reason={action['finish_reason']}；"
                f"字符数={action['characters']} ---",
                flush=True,
            )
        return response

    result = execute_unified_request(
        provider_name,
        model,
        args.question,
        payload,
        providers=PROVIDERS,
        complete=complete,
    )
    result["diagnostic_model_actions"] = model_actions
    result["diagnostic_elapsed_seconds"] = round(time.perf_counter() - started, 3)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    print(f"完成：{time.perf_counter() - started:.2f}s，状态={result.get('status')}，"
          f"模型调用={result.get('model_calls', 0)}，代码执行={result.get('code_execution_count', 0)}")
    for index, step in enumerate(result.get("agent_trace") or [], 1):
        tools_used = ", ".join(
            str(call.get("api")) + ("（缓存）" if call.get("cache_hit") else "")
            for call in step.get("calls") or []
        )
        suffix = f"；工具：{tools_used}" if tools_used else ""
        print(f"{index}. {step.get('summary') or step.get('action')}{suffix}")
        if step.get("error"):
            print("   错误：" + _brief(step["error"]))
    print("\nAgent 回答：\n" + str(result.get("answer", "")))


if __name__ == "__main__":
    main()
