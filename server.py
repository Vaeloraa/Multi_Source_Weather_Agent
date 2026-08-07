from __future__ import annotations

import json
import io
import os
import re
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from sevir_world.repositories.ensemble import get_cascast_display_frame
from sevir_world.repositories.uploads import (
    get_uploaded_display_frame,
    get_uploaded_sequence_display_frame,
    list_uploaded_records,
    register_uploaded_vil,
    register_uploaded_vil_sequence,
)
from sevir_world.services.status import runtime_status
from sevir_world.services.tool_gateway import TOOLS, execute_tool
from sevir_world.direct_agent import run_direct_agent


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
ARTIFACT_ROOT = ROOT / "data" / "artifacts"
FORECAST_TOOL_NAMES = {
    "run_nowcast",
    "forecast_hazards",
    "analyze_ensemble_uncertainty",
    "evaluate_cascast_dataset",
    "analyze_uploaded_vil",
    "analyze_uploaded_vil_sequence",
    "get_sevir_event_metadata",
    "load_sevir_vil_event",
    "search_sevir_catalog",
    "historical_sevir_statistics",
    "geolocate_sevir_event",
    "get_noaa_storm_reference",
    "compare_event_with_history",
}
AGENT_TOOLS = [
    tool for tool in TOOLS if tool.get("function", {}).get("name") in FORECAST_TOOL_NAMES
]
FORECAST_FAMILY_TOOLS = {
    "run_nowcast",
    "forecast_hazards",
    "analyze_ensemble_uncertainty",
}

# Hazard policy is owned by the backend. The UI selects a named level; numeric
# thresholds remain accepted as a compatibility transport for existing clients.
HAZARD_PROFILES = {
    "noticeable": {"label": "明显回波", "threshold": 74.0, "minimum_region_pixels": 16, "minimum_duration_minutes": 5},
    "strong": {"label": "较强回波", "threshold": 133.0, "minimum_region_pixels": 16, "minimum_duration_minutes": 5},
    "very_strong": {"label": "很强回波", "threshold": 160.0, "minimum_region_pixels": 16, "minimum_duration_minutes": 5},
    "extreme": {"label": "极强回波", "threshold": 181.0, "minimum_region_pixels": 16, "minimum_duration_minutes": 5},
    "extreme_candidate": {"label": "极端强回波候选", "threshold": 219.0, "minimum_region_pixels": 16, "minimum_duration_minutes": 5},
}
DEFAULT_HAZARD_LEVEL = "strong"


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    "kimi": {
        "label": "Kimi",
        "base_url": os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        "api_key": os.getenv("KIMI_API_KEY", ""),
        "model": os.getenv("KIMI_MODEL", "kimi-k2.5"),
    },
    "qwen": {
        "label": "Qwen",
        "base_url": os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "model": os.getenv("QWEN_MODEL", "qwen-plus"),
    },
}


SYSTEM_PROMPT = """你是一个以CasCast为核心、面向SEVIR强回波临近预报的气象智能体。
当前Demo已接入10个SEVIR测试事件的真实CasCast推理结果；每个事件包含13帧历史观测、10路ensemble、未来12帧（60分钟）VIL预测和严格对齐的12帧ground truth。
你必须根据问题调用提供的工具，不得编造数据、模型指标或未执行的预测。
可以使用analyze_ensemble_uncertainty报告pixel-wise标准差、超阈值概率和storm-object差异；必须说明ensemble分歧不等于真实误差。
当用户询问全部10个事件的整体表现、多阈值评分、逐时效退化、与Persistence比较、不确定性是否可信或storm-object总体评价时，调用evaluate_cascast_dataset；不要把10事件结果写成完整SEVIR测试集结论。
当问题确实同时要求预测、定位、CATALOG查询或历史比较时，可以连续调用最多3个互补工具，再综合回答。
精确事件详情使用get_sevir_event_metadata；只有用户明确要求读取本地HDF5真实像素或49帧统计时才使用load_sevir_vil_event。
NOAA关联使用get_noaa_storm_reference；随机R类事件通常没有NOAA编号，不得据此声称无灾害。
当问题要求“预测并分析危险区域、位置或持续时间”时，直接调用forecast_hazards；它内部已经完成nowcast，禁止先调用run_nowcast再调用forecast_hazards。
工具选择规则：只询问未来VIL数值、逐帧图像或变化趋势时选择run_nowcast；明确询问危险区域、H1/H2/H3、位置、面积、开始时间或持续时间时选择forecast_hazards；询问ensemble成员分歧、概率或不确定性时选择analyze_ensemble_uncertainty。这三个预测家族工具功能重叠，同一问题最多调用其中一个；它们可以和定位、CATALOG或历史比较工具组合。
可以根据工具返回的model_metrics报告当前事件的CSI、POD、FAR、MSE和MAE；必须说明它们是单事件结果，不能代表完整测试集性能。
VIL阈值是0到255范围内的缩放像素值，只能作为强回波和潜在降雨危险的代理指标，不能写成dBZ、降雨量或官方预警。
坐标由SEVIR事件边界线性映射得到，只能称为近似坐标。
当前只支持未来60分钟的单次预测；2至3小时滚动预测尚未执行。
当用户上传单张SEVIR VIL图片时，只能分析当前图像的强回波强度、面积和候选hazard区域；禁止声称完成未来预测、移动趋势或持续时间分析。
当用户上传13帧连续VIL时，可以分析过去60分钟演变并运行Persistence基线，但不得称为CasCast；任意序列CasCast仍缺EarthFormer现场确定性预测与latent压缩链路。
历史统计和事件检索来自SEVIR CATALOG metadata。必须区分catalog记录级统计与逐像素VIL分析；事件数量不能直接写成灾害次数，catalog data_max分位数不能写成灾害概率。
Geolocator当前只提供SEVIR覆盖框和近似中心，不提供行政区名称。
回答应说明事件、模型、预测时效、危险区域、持续时间和结论限制，并使用简洁中文。"""


def selected_hazard_profile(payload: dict[str, Any]) -> dict[str, Any]:
    level = str(payload.get("hazard_level", "")).strip()
    if level:
        if level not in HAZARD_PROFILES:
            raise ValueError(f"未知强回波等级：{level}。")
        return HAZARD_PROFILES[level]

    # Compatibility with clients that previously sent the numeric threshold.
    raw_threshold = payload.get("threshold")
    if raw_threshold not in {None, ""}:
        threshold = float(raw_threshold)
        for profile in HAZARD_PROFILES.values():
            if profile["threshold"] == threshold:
                return profile
        raise ValueError("VIL阈值必须使用系统预设等级。")
    return HAZARD_PROFILES[DEFAULT_HAZARD_LEVEL]


def selected_threshold(payload: dict[str, Any]) -> float:
    return float(selected_hazard_profile(payload)["threshold"])


def normalize_base_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def post_chat_completion(
    provider: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    request = urllib.request.Request(
        normalize_base_url(provider["base_url"]),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API返回HTTP {error.code}：{detail[:600]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"无法连接模型API：{error.reason}") from error


def parse_tool_call(message: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    calls = message.get("tool_calls") or []
    if not calls:
        raise RuntimeError("模型没有选择SEVIR工具，请把任务和事件编号描述得更明确。")
    call = calls[0]
    function = call.get("function", {})
    try:
        arguments = json.loads(function.get("arguments", "{}"))
    except json.JSONDecodeError as error:
        raise RuntimeError("模型返回了无法解析的工具参数。") from error
    return function.get("name", ""), arguments, call.get("id", "tool_call_1")


def bound_arguments_for_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    event_id = str(payload.get("event_id", "R19100105107994")).strip()
    model_name = str(payload.get("weather_model", "cascast"))
    hazard_profile = selected_hazard_profile(payload)
    threshold = float(hazard_profile["threshold"])
    minimum_region_pixels = int(hazard_profile["minimum_region_pixels"])
    lead_minutes = int(payload.get("lead_minutes", 60))
    upload_id = str(payload.get("upload_id", "")).strip()
    sequence_upload_id = str(payload.get("sequence_upload_id", "")).strip()
    if tool_name == "run_nowcast":
        return {"event_id": event_id, "model_name": model_name, "input_frames": 13, "forecast_frames": lead_minutes // 5, "threshold": threshold}
    if tool_name == "forecast_hazards":
        return {"event_id": event_id, "model_name": model_name, "lead_minutes": lead_minutes, "threshold": threshold, "minimum_region_pixels": minimum_region_pixels}
    if tool_name == "analyze_ensemble_uncertainty":
        return {"event_id": event_id, "threshold": threshold, "minimum_region_pixels": minimum_region_pixels}
    if tool_name == "evaluate_cascast_dataset":
        return {"primary_threshold": threshold}
    if tool_name == "analyze_uploaded_vil":
        if not upload_id:
            raise ValueError("这个问题需要图片数据，请先选择官方SEVIR示例或上传一张384×384灰度PNG。")
        return {"upload_id": upload_id, "threshold": threshold, "minimum_region_pixels": minimum_region_pixels}
    if tool_name == "analyze_uploaded_vil_sequence":
        if not sequence_upload_id:
            raise ValueError("这个问题需要13帧连续VIL，请先上传完整序列。")
        return {"upload_id": sequence_upload_id, "threshold": threshold, "minimum_region_pixels": minimum_region_pixels}
    if tool_name in {
        "get_sevir_event_metadata",
        "load_sevir_vil_event",
        "geolocate_sevir_event",
        "get_noaa_storm_reference",
        "compare_event_with_history",
    }:
        return {"event_id": event_id}
    return {}


def heuristic_tool_for_question(question: str, payload: dict[str, Any]) -> str:
    text = question.lower()
    if str(payload.get("sequence_upload_id", "")).strip() and any(
        word in text for word in ("上传", "序列", "13帧", "过去", "演变", "趋势")
    ):
        return "analyze_uploaded_vil_sequence"
    if any(word in text for word in ("上传", "这张图", "图片", "单张")):
        return "analyze_uploaded_vil"
    if any(word in text for word in ("历史分位", "历史相比", "历史比较", "同类事件")):
        return "compare_event_with_history"
    if any(word in text for word in ("noaa", "episode_id", "episode id", "灾害记录", "灾情记录")):
        return "get_noaa_storm_reference"
    if any(word in text for word in ("hdf5", "49帧", "真实像素", "逐帧真实", "读取像素")):
        return "load_sevir_vil_event"
    if any(word in text for word in ("minute_offsets", "时间偏移", "投影参数", "完整metadata", "详细元数据")):
        return "get_sevir_event_metadata"
    if any(word in text for word in ("经纬度", "坐标", "覆盖范围", "空间范围", "在哪里")) and not any(
        word in text for word in ("未来", "预测", "hazard", "危险")
    ):
        return "geolocate_sevir_event"
    if any(word in text for word in ("统计", "storm type", "事件类型", "月份", "年份", "多少条", "分布")):
        return "historical_sevir_statistics"
    if any(word in text for word in ("catalog", "metadata", "元数据", "文件位置", "查询事件")):
        return "search_sevir_catalog"
    if any(word in text for word in ("ensemble", "不确定", "置信", "分歧", "概率图", "成员差异")):
        if any(word in text for word in ("整体", "全部", "10个", "十个", "测试集", "可靠", "校准", "表现")):
            return "evaluate_cascast_dataset"
        return "analyze_ensemble_uncertainty"
    if any(word in text for word in ("整体表现", "全部事件", "测试集评价", "多阈值", "逐时效", "基线对比", "persistence比较")):
        return "evaluate_cascast_dataset"
    if any(word in text for word in ("hazard", "危险区域", "持续多久", "持续时间", "哪些地方", "强回波区域")):
        return "forecast_hazards"
    return "run_nowcast"


def natural_language_context(payload: dict[str, Any]) -> str:
    context = {
        "selected_data_source": str(payload.get("data_source", "cascast")),
        "active_event_id": str(payload.get("event_id", "R19100105107994")).strip(),
        "weather_model": str(payload.get("weather_model", "cascast")),
        "forecast_horizon_minutes": int(payload.get("lead_minutes", 60)),
        "vil_threshold": selected_threshold(payload),
        "uploaded_image_available": bool(str(payload.get("upload_id", "")).strip()),
        "uploaded_13_frame_sequence_available": bool(str(payload.get("sequence_upload_id", "")).strip()),
    }
    return (
        "\n\n当前网页数据上下文如下。你应根据用户自然语言自行选择最合适的工具；"
        "涉及当前事件、模型、时效、阈值或上传图片时，必须使用这些上下文值，不得自行编造："
        + json.dumps(context, ensure_ascii=False)
    )


def run_auto_agent(provider_name: str, model: str, question: str, payload: dict[str, Any]) -> dict[str, Any]:
    provider = PROVIDERS[provider_name]
    if not provider["api_key"]:
        raise RuntimeError(f"尚未配置{provider['label']} API Key。")
    reflective_instruction = (
        "\n\n这是Reflective模式：先规划所需工具，执行后复核结果是否足以回答；"
        "必要时调用补充工具，最终明确区分数据事实、推断和能力限制。"
        if str(payload.get("mode", "api")) == "reflective"
        else ""
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question + natural_language_context(payload) + reflective_instruction},
    ]
    registered = {tool["function"]["name"] for tool in AGENT_TOOLS}
    tool_runs: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    usages: list[dict[str, Any]] = []
    route_warning = ""
    final_answer = ""

    for step in range(3):
        try:
            response = post_chat_completion(provider, model, messages, AGENT_TOOLS, tool_choice="auto")
            usages.append(response.get("usage", {}))
            assistant_message = response["choices"][0]["message"]
            if not assistant_message.get("tool_calls"):
                if not tool_runs:
                    raise RuntimeError("模型没有先调用可验证的SEVIR工具。")
                final_answer = assistant_message.get("content", "")
                break
            tool_name, arguments, call_id = parse_tool_call(assistant_message)
            if tool_name not in registered:
                raise RuntimeError(f"模型选择了未注册工具：{tool_name}")
        except (RuntimeError, KeyError, IndexError) as error:
            if tool_runs:
                route_warning = f"第{step + 1}步工具规划停止：{error}"
                break
            raise RuntimeError(f"Qwen没有完成有效工具选择：{error}") from error

        prior_tool_names = {item["tool_name"] for item in tool_runs}
        if tool_name in FORECAST_FAMILY_TOOLS and prior_tool_names.intersection(FORECAST_FAMILY_TOOLS):
            route_warning = (
                "模型请求了功能重叠的预测工具；系统保留第一次预测结果并停止重复执行。"
            )
            break

        arguments.update(bound_arguments_for_tool(tool_name, payload))
        signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
        if signature in seen_calls:
            route_warning = "模型请求了重复工具调用，系统已停止重复执行。"
            break
        seen_calls.add(signature)
        tool_result = execute_tool(tool_name, arguments)
        tool_runs.append(
            {
                "step": step + 1,
                "tool_name": tool_name,
                "tool_arguments": arguments,
                "tool_result": tool_result,
            }
        )
        if call_id.startswith("local_route") or tool_result.get("status", "ok").startswith("awaiting_"):
            break
        messages.extend(
            [
                assistant_message,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                },
            ]
        )

    if not tool_runs:
        raise RuntimeError("Agent没有完成任何可验证的工具调用。")

    priority = {
        "evaluate_cascast_dataset": 5,
        "analyze_ensemble_uncertainty": 4,
        "forecast_hazards": 3,
        "run_nowcast": 2,
    }
    primary = max(tool_runs, key=lambda item: priority.get(item["tool_name"], 1))
    if not final_answer and not primary["tool_result"].get("status", "ok").startswith("awaiting_"):
        try:
            synthesis = post_chat_completion(provider, model, messages)
            usages.append(synthesis.get("usage", {}))
            final_answer = synthesis["choices"][0]["message"].get("content", "")
        except (RuntimeError, KeyError, IndexError) as error:
            route_warning = f"{route_warning} 综合解释未完成：{error}".strip()
    if not final_answer:
        final_answer = local_answer(primary["tool_name"], primary["tool_result"])

    return {
        "answer": final_answer,
        "tool_name": primary["tool_name"],
        "tool_arguments": primary["tool_arguments"],
        "tool_result": primary["tool_result"],
        "tool_runs": tool_runs,
        "agent_warning": route_warning,
        "route_mode": "reflective_multi_tool" if len(tool_runs) > 1 else "natural_language_auto",
        "usage": {"calls": usages},
    }


def run_api_agent(
    provider_name: str,
    model: str,
    question: str,
    required_tool_name: str,
    required_tool_arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = PROVIDERS[provider_name]
    if not provider["api_key"]:
        raise RuntimeError(f"尚未配置{provider['label']} API Key。")
    bound_arguments = required_tool_arguments or {}
    task_context = (
        "\n\n本次任务已经由系统绑定以下数据和参数，调用工具时必须原样使用："
        + json.dumps(bound_arguments, ensure_ascii=False)
        if bound_arguments
        else ""
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question + task_context},
    ]
    selected_tools = [
        tool
        for tool in AGENT_TOOLS
        if tool.get("function", {}).get("name") == required_tool_name
    ]
    if not selected_tools:
        raise RuntimeError(f"Agent工具未注册：{required_tool_name}")
    try:
        first = post_chat_completion(
            provider,
            model,
            messages,
            selected_tools,
            tool_choice="auto",
        )
        assistant_message = first["choices"][0]["message"]
        tool_name, arguments, call_id = parse_tool_call(assistant_message)
        arguments.update(bound_arguments)
        tool_result = execute_tool(tool_name, arguments)
    except (RuntimeError, KeyError, IndexError) as error:
        tool_name = required_tool_name
        arguments = dict(bound_arguments)
        tool_result = execute_tool(tool_name, arguments)
        return {
            "answer": local_answer(tool_name, tool_result),
            "tool_name": tool_name,
            "tool_arguments": arguments,
            "tool_result": tool_result,
            "agent_warning": f"Qwen工具调用未完成，已降级为真实本地工具结果：{error}",
            "usage": {"first_call": {}, "second_call": {}},
        }
    if tool_result.get("status", "ok").startswith("awaiting_"):
        return {
            "answer": local_answer(tool_name, tool_result),
            "tool_name": tool_name,
            "tool_arguments": arguments,
            "tool_result": tool_result,
            "usage": {
                "first_call": first.get("usage", {}),
                "second_call": {},
            },
        }
    messages.extend(
        [
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            },
        ]
    )
    try:
        second = post_chat_completion(provider, model, messages)
        answer = second["choices"][0]["message"].get("content", "")
        warning = ""
    except (RuntimeError, KeyError, IndexError) as error:
        second = {}
        answer = local_answer(tool_name, tool_result)
        warning = f"Qwen结果解释未完成，已返回真实工具结果：{error}"
    return {
        "answer": answer,
        "tool_name": tool_name,
        "tool_arguments": arguments,
        "tool_result": tool_result,
        "usage": {
            "first_call": first.get("usage", {}),
            "second_call": second.get("usage", {}),
        },
        "agent_warning": warning,
    }


def local_task_arguments(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    task = str(payload.get("task", "multi_hour_forecast"))
    event_id = str(payload.get("event_id", "R19100105107994")).strip()
    model_name = str(payload.get("weather_model", "cascast"))
    threshold = selected_threshold(payload)
    lead_minutes = int(payload.get("lead_minutes", 60))
    if task == "uploaded_image_analysis":
        upload_id = str(payload.get("upload_id", "")).strip()
        if not upload_id:
            raise ValueError("请先上传一张384×384的SEVIR VIL图片。")
        return "analyze_uploaded_vil", {
            "upload_id": upload_id,
            "threshold": threshold,
            "minimum_region_pixels": 16,
        }
    if task == "uploaded_sequence_analysis":
        upload_id = str(payload.get("sequence_upload_id", "")).strip()
        if not upload_id:
            raise ValueError("请先上传13张连续的384×384 SEVIR VIL灰度图。")
        return "analyze_uploaded_vil_sequence", {
            "upload_id": upload_id,
            "threshold": threshold,
            "minimum_region_pixels": 16,
        }
    if task == "catalog_search":
        return "search_sevir_catalog", {}
    if task == "historical_statistics":
        return "historical_sevir_statistics", {}
    if task == "event_geolocation":
        return "geolocate_sevir_event", {"event_id": event_id}
    if task == "event_history_comparison":
        return "compare_event_with_history", {"event_id": event_id}
    if task == "multi_hour_forecast":
        return "run_nowcast", {
            "event_id": event_id,
            "model_name": model_name,
            "input_frames": 13,
            "forecast_frames": lead_minutes // 5,
            "threshold": threshold,
        }
    if task == "ensemble_uncertainty":
        return "analyze_ensemble_uncertainty", {
            "event_id": event_id,
            "threshold": threshold,
            "minimum_region_pixels": 16,
        }
    if task == "dataset_evaluation":
        return "evaluate_cascast_dataset", {"primary_threshold": threshold}
    if task in {"hazard_locations", "hazard_duration", "hazard_evolution"}:
        return "forecast_hazards", {
            "event_id": event_id,
            "model_name": model_name,
            "lead_minutes": lead_minutes,
            "threshold": threshold,
            "minimum_region_pixels": 16,
        }
    if task == "satellite_to_radar":
        return "run_satellite_to_radar", {"event_id": event_id}
    if task == "simulator":
        return "run_simulator", {"event_id": event_id, "simulator_name": "jcm"}
    raise ValueError(f"未知任务类型：{task}")


def local_answer(name: str, result: dict[str, Any]) -> str:
    if result.get("status", "ok").startswith("awaiting_"):
        missing = "、".join(result.get("required_assets", []))
        return (
            "本任务要求使用CasCast，但当前无法执行真实预测，系统已停止计算且没有用"
            f"persistence替代。缺少：{missing}。"
        )
    if name == "get_event_summary":
        return (
            f"事件{result['event_id']}包含{result['frame_count']}帧真实VIL数据，"
            f"空间尺寸为{result['spatial_shape'][0]}×{result['spatial_shape'][1]}，"
            f"最大VIL为{result['max_vil']}。"
        )
    if name == "analyze_hazard":
        return (
            f"观测序列中最长连续强回波时段约"
            f"{result['longest_active_duration_minutes']}分钟。"
            "该结果不是未来预报或官方灾害定级。"
        )
    if name == "analyze_uploaded_vil":
        return (
            f"已分析上传图片{result['filename']}：最大像素值为{result['max_vil']}，"
            f"阈值以上面积占比为{result['exceeding_area_ratio'] * 100:.2f}%，"
            f"筛查到{result['hazard_region_count']}个候选强回波区域。"
            "这是单张当前图像分析，不是未来预测。"
        )
    if name == "analyze_uploaded_vil_sequence":
        tracks = result["historical_storm_tracks"]
        return (
            f"已分析上传的13帧连续VIL历史序列，检测到{tracks['track_count']}条连通区域轨迹，"
            f"最长连续轨迹为{tracks['longest_track_duration_minutes']}分钟。"
            "未来60分钟图像使用最后观测帧持续不变的Persistence基线，不是CasCast预测。"
        )
    if name == "search_sevir_catalog":
        return (
            f"已查询SEVIR VIL catalog，共找到{result['total_matches']}条匹配记录，"
            f"本次返回{result['returned_count']}条。"
        )
    if name == "get_sevir_event_metadata":
        record = result["record"]
        return (
            f"已精确读取事件{record['sevir_id']}的VIL metadata，共有"
            f"{len(record['minute_offsets'])}个时间偏移；"
            f"本地HDF5可用状态为{record['local_hdf5_available']}。"
        )
    if name == "load_sevir_vil_event":
        if not result.get("pixel_data_loaded"):
            return (
                f"已定位事件{result['event_id']}，但本地缺少官方HDF5文件"
                f"{result['required_hdf5']}，因此尚未读取逐帧像素。"
            )
        return (
            f"已从官方SEVIR HDF5读取事件{result['event_id']}的真实像素，"
            f"本次读取形状为{result['loaded_shape']}，并生成逐帧时间和VIL统计。"
        )
    if name == "get_noaa_storm_reference":
        if result["noaa_linked"]:
            return (
                f"事件{result['sevir_id']}关联NOAA event {result['noaa_event_id']}、"
                f"episode {result['noaa_episode_id']}，可通过官方详情链接核对。"
            )
        return f"事件{result['sevir_id']}在SEVIR CATALOG中没有NOAA关联编号。"
    if name == "historical_sevir_statistics":
        summary = result.get("data_max_summary") or {}
        return (
            f"已完成{result['record_count']}条SEVIR VIL catalog记录的历史统计。"
            f"catalog data_max中位数为{summary.get('p50', '未知')}，"
            f"95分位数为{summary.get('p95', '未知')}。"
            "这是metadata统计，不是逐帧降雨量分析。"
        )
    if name == "geolocate_sevir_event":
        center = result["approx_center_latlon"]
        return (
            f"事件{result['event_id']}的覆盖区域近似中心为"
            f"{center[0]}°N, {abs(center[1])}°W。"
            "这是SEVIR覆盖框中心，不是行政区定位。"
        )
    if name == "compare_event_with_history":
        return (
            f"事件{result['event_id']}的catalog data_max为{result['catalog_data_max']}，"
            f"在{result['reference_group']}参考组中约位于第"
            f"{result['data_max_percentile']}百分位。该分位不是灾害概率。"
        )
    if name == "run_nowcast":
        if result.get("model_name") == "cascast":
            metrics = result["model_metrics"]
            return (
                f"已读取事件{result['event_id']}的真实CasCast输出，"
                f"包含未来{result['lead_minutes']}分钟、{result['forecast_frames']}帧VIL预测。"
                f"相对于对应ground truth，CSI@VIL{metrics['metric_threshold']:g}="
                f"{metrics['csi_at_threshold']}，MSE={metrics['mse_raw_vil']}。"
                "这些指标只对应当前事件。"
            )
        if result.get("ground_truth_available") is False:
            return (
                f"已使用所选事件相同的13帧历史输入生成Persistence未来"
                f"{result['forecast_frames'] * 5}分钟基线。"
                "当前没有ground truth，因此不报告CSI或MSE，也不能声称与CasCast完成准确率对比。"
            )
        return (
            f"已使用{result['model_name']}预测未来{result['forecast_frames'] * 5}分钟；"
            f"CSI@VIL{result['metric_threshold']:g}={result['csi_at_threshold']}，"
            f"归一化MSE={result['mse_normalized_0_1']}。"
            "这是基线结果，不是CasCast结果。"
        )
    if name == "analyze_ensemble_uncertainty":
        pixel = result["pixel_uncertainty"]
        objects = result["storm_object_uncertainty"]
        duration = result["member_active_duration_minutes"]
        return (
            f"事件{result['event_id']}的10路CasCast ensemble已完成不确定性分析。"
            f"逐像素标准差均值为{pixel['mean_std']}，95分位为{pixel['p95_std']}；"
            f"有{objects['members_with_object']}路成员检测到阈值以上storm object，"
            f"其中心位置离散度为{objects['centroid_spread_pixels']}像素。"
            f"不同成员的强回波持续时间范围为{duration['min']}至{duration['max']}分钟。"
            "这些数值反映ensemble分歧，不代表真实预测误差。"
        )
    if name == "evaluate_cascast_dataset":
        threshold_key = str(int(float(result["primary_threshold"])))
        cascast = result["aggregate_metrics"]["cascast_ensemble_mean"][threshold_key]
        persistence = result["aggregate_metrics"]["persistence"][threshold_key]
        validation = result["ensemble_validation"]
        objects = result["storm_object_evaluation"]
        return (
            f"已完成全部{result['event_count']}个本地SEVIR测试事件的聚合评价。"
            f"在VIL阈值{result['primary_threshold']:g}下，CasCast ensemble mean的"
            f"CSI为{cascast['csi']}、RMSE为{cascast['rmse_raw_vil']}，"
            f"Persistence的CSI为{persistence['csi']}、RMSE为{persistence['rmse_raw_vil']}。"
            f"像素标准差与绝对误差的相关系数为{validation['pixel_std_absolute_error_pearson']}，"
            f"storm-object CSI为{objects['object_csi']}。"
            "这是10个事件的实验结果，不是完整SEVIR测试集复现。"
        )
    if name == "forecast_hazards":
        if not result["hazard_detected"]:
            if result.get("threshold_exceeded"):
                return (
                    f"未来{result['lead_minutes']}分钟存在零散像素达到VIL阈值"
                    f"{result['threshold']}，但单帧最多只有"
                    f"{result['maximum_exceeding_pixel_count']}个超阈值像素，"
                    f"未达到{result['hazard_area_minimum_pixels']}像素的区域判据，"
                    "因此不判定为Hazard区域。"
                    f"任意像素超阈值的最长连续时段为"
                    f"{result['any_pixel_exceedance_longest_consecutive_minutes']}分钟。"
                )
            return f"未来{result['lead_minutes']}分钟没有像素达到VIL阈值{result['threshold']}。"
        region_text = "；".join(
            f"{region['region_id']}中心约为{region.get('approx_centroid_latlon', '未知')}"
            for region in result["hazard_regions"]
        )
        lower_bound = "至少" if result["duration_is_lower_bound"] else "约"
        result_note = (
            "这是已保存的真实CasCast预测分析。"
            if result.get("model_name") == "cascast"
            else "当前使用persistence基线，不是CasCast结果。"
        )
        return (
            f"{result['model_name']}在未来{result['lead_minutes']}分钟筛查到"
            f"{result['hazard_region_count']}个强回波区域：{region_text}。"
            f"强回波在预测窗口内持续{lower_bound}"
            f"{result['strong_echo_duration_minutes_within_horizon']}分钟。"
            f"{result_note}"
        )
    return "工具已执行。"


def run_local_task(payload: dict[str, Any]) -> dict[str, Any]:
    if str(payload.get("task", "auto")) == "auto":
        name = heuristic_tool_for_question(str(payload.get("question", "")), payload)
        arguments = bound_arguments_for_tool(name, payload)
    else:
        name, arguments = local_task_arguments(payload)
    result = execute_tool(name, arguments)
    return {
        "answer": local_answer(name, result),
        "tool_name": name,
        "tool_arguments": arguments,
        "tool_result": result,
        "usage": {},
    }


def run_direct_mode(
    provider_name: str, model: str, question: str, payload: dict[str, Any], *, max_attempts: int = 3
) -> dict[str, Any]:
    """Run a Zephyrus code agent through the SEVIRWorld safety boundary."""
    provider = PROVIDERS[provider_name]
    context = {
        "event_id": str(payload.get("event_id", "")).strip(),
        "data_source": str(payload.get("data_source", "cascast")),
        "weather_model": str(payload.get("weather_model", "cascast")),
        "lead_minutes": int(payload.get("lead_minutes", 60)),
        "threshold": selected_threshold(payload),
        "upload_id": str(payload.get("upload_id", "")).strip(),
        "sequence_upload_id": str(payload.get("sequence_upload_id", "")).strip(),
    }
    data_descs = payload.get("data_descs")
    if data_descs is not None and not isinstance(data_descs, list):
        raise ValueError("data_descs must be a list")
    selected_tools = payload.get("selected_tools")
    if selected_tools is not None and not isinstance(selected_tools, list):
        raise ValueError("selected_tools must be a list")

    def complete(current_provider: dict[str, str], current_model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return post_chat_completion(current_provider, current_model, messages)

    result = run_direct_agent(
        provider,
        model,
        question,
        context,
        complete,
        max_attempts=max_attempts,
        data_descs=data_descs,
        selected_tools=selected_tools,
    )
    result["route_mode"] = "zephyrus_reflective" if max_attempts > 1 else "zephyrus_direct"
    return result


def render_vil_png(frame: np.ndarray) -> bytes:
    """Render the original VIL grid with one consistent scientific palette."""
    values = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 254.0)
    stops = np.asarray([0, 40, 74, 133, 160, 181, 219, 254], dtype=np.float32)
    colors = np.asarray(
        [
            [7, 15, 28],
            [25, 74, 130],
            [31, 158, 184],
            [76, 190, 108],
            [234, 210, 73],
            [239, 132, 55],
            [207, 49, 65],
            [247, 238, 246],
        ],
        dtype=np.float32,
    )
    rgb = np.stack(
        [np.interp(values, stops, colors[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_uncertainty_png(frame: np.ndarray, kind: str) -> bytes:
    values = np.asarray(frame, dtype=np.float32)
    if kind == "probability":
        normalized = np.clip(values, 0.0, 1.0)
        stops = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        colors = np.asarray(
            [[8, 18, 38], [24, 78, 138], [24, 160, 170], [231, 202, 64], [235, 103, 47], [190, 28, 58]],
            dtype=np.float32,
        )
    elif kind == "duration":
        normalized = np.clip(values, 0.0, 60.0)
        stops = np.asarray([0.0, 5.0, 15.0, 30.0, 45.0, 60.0], dtype=np.float32)
        colors = np.asarray(
            [[8, 18, 38], [24, 78, 138], [24, 160, 170], [231, 202, 64], [235, 103, 47], [190, 28, 58]],
            dtype=np.float32,
        )
    else:
        normalized = np.clip(values, 0.0, 50.0)
        stops = np.asarray([0.0, 5.0, 10.0, 20.0, 35.0, 50.0], dtype=np.float32)
        colors = np.asarray(
            [[9, 18, 35], [32, 72, 142], [70, 119, 190], [135, 83, 178], [218, 92, 112], [251, 190, 70]],
            dtype=np.float32,
        )
    rgb = np.stack(
        [np.interp(normalized, stops, colors[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {format % args}")

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_png(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            providers = {
                key: {
                    "label": value["label"],
                    "model": value["model"],
                    "configured": bool(value["api_key"]),
                }
                for key, value in PROVIDERS.items()
            }
            self.send_json({"providers": providers, "runtime": runtime_status()})
            return
        artifact_match = re.fullmatch(r"/api/artifacts/([0-9a-f]{32})\.png", parsed.path)
        if artifact_match:
            artifact_path = ARTIFACT_ROOT / f"{artifact_match.group(1)}.png"
            if not artifact_path.is_file():
                self.send_json({"error": "Artifact not found"}, 404)
                return
            self.send_png(artifact_path.read_bytes())
            return
        if parsed.path == "/api/cascast-frame":
            try:
                query = parse_qs(parsed.query)
                event_id = str(query.get("event_id", [""])[0]).strip()
                series = str(query.get("series", ["forecast"])[0])
                model_name = str(query.get("model_name", ["cascast"])[0])
                frame_index = int(query.get("index", ["0"])[0])
                lead_index = query.get("lead", [None])[0]
                lead_index = int(lead_index) if lead_index not in {None, ""} else None
                threshold = float(query.get("threshold", ["133"])[0])
                frame, _ = get_cascast_display_frame(
                    event_id=event_id,
                    series=series,
                    frame_index=frame_index,
                    model_name=model_name,
                    threshold=threshold,
                    lead_index=lead_index,
                )
                if series == "uncertainty_std":
                    self.send_png(render_uncertainty_png(frame, "std"))
                elif series in {"uncertainty_probability", "hazard_probability_any"}:
                    self.send_png(render_uncertainty_png(frame, "probability"))
                elif series == "hazard_duration_mean":
                    self.send_png(render_uncertainty_png(frame, "duration"))
                else:
                    self.send_png(render_vil_png(frame))
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        if parsed.path == "/api/uploaded-frame":
            try:
                query = parse_qs(parsed.query)
                upload_id = str(query.get("upload_id", [""])[0]).strip()
                self.send_png(render_vil_png(get_uploaded_display_frame(upload_id)))
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        if parsed.path == "/api/uploads":
            self.send_json(list_uploaded_records())
            return
        if parsed.path == "/api/uploaded-sequence-frame":
            try:
                query = parse_qs(parsed.query)
                upload_id = str(query.get("upload_id", [""])[0]).strip()
                series = str(query.get("series", ["history"])[0])
                frame_index = int(query.get("index", ["0"])[0])
                frame = get_uploaded_sequence_display_frame(upload_id, series, frame_index)
                self.send_png(render_vil_png(frame))
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path not in {"/api/run", "/api/upload", "/api/upload-sequence"}:
            self.send_json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 5_000_000:
                raise ValueError("请求过大；上传数据处理后不得超过5MB。")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/upload":
                result = register_uploaded_vil(
                    filename=str(payload.get("filename", "uploaded_vil.png")),
                    width=int(payload.get("width", 0)),
                    height=int(payload.get("height", 0)),
                    pixels_base64=str(payload.get("pixels_base64", "")),
                    grayscale_source=bool(payload.get("grayscale_source", False)),
                )
                self.send_json(result)
                return
            if self.path == "/api/upload-sequence":
                result = register_uploaded_vil_sequence(list(payload.get("frames", [])))
                self.send_json(result)
                return
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("问题不能为空。")
            selected_threshold(payload)
            if str(payload.get("data_source", "cascast")) == "upload" and not str(
                payload.get("upload_id", "")
            ).strip():
                raise ValueError("请先上传一张384×384的SEVIR VIL图片。")
            if str(payload.get("data_source", "cascast")) == "upload_sequence" and not str(
                payload.get("sequence_upload_id", "")
            ).strip():
                raise ValueError("请先上传13张连续的384×384 SEVIR VIL图片。")
            started = time.perf_counter()
            mode = str(payload.get("mode", "local"))
            provider_name = str(payload.get("provider", "qwen"))
            if provider_name not in PROVIDERS:
                raise ValueError("不支持的大模型提供方。")
            if mode == "direct":
                model = str(payload.get("model") or PROVIDERS[provider_name]["model"])
                result = run_direct_mode(provider_name, model, question, payload, max_attempts=1)
            elif mode == "reflective":
                model = str(payload.get("model") or PROVIDERS[provider_name]["model"])
                result = run_direct_mode(provider_name, model, question, payload, max_attempts=3)
            elif mode == "api":
                model = str(payload.get("model") or PROVIDERS[provider_name]["model"])
                if str(payload.get("task", "auto")) == "auto":
                    result = run_auto_agent(provider_name, model, question, payload)
                else:
                    required_tool_name, required_tool_arguments = local_task_arguments(payload)
                    result = run_api_agent(provider_name, model, question, required_tool_name, required_tool_arguments)
            else:
                result = run_local_task(payload)
            result.update(
                {
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "mode": mode,
                    "provider": provider_name,
                }
            )
            self.send_json(result)
        except (ValueError, RuntimeError, KeyError, IndexError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.send_json({"error": f"服务器内部错误：{error}"}, 500)


def main() -> None:
    host = os.getenv("SEVIR_DEMO_HOST", "127.0.0.1")
    port = int(os.getenv("SEVIR_DEMO_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"SEVIR Hazard Agent: http://{host}:{port}")
    print("按Ctrl+C停止服务。")
    server.serve_forever()


if __name__ == "__main__":
    main()
