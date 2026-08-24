from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "R19062708588472"


@dataclass
class Check:
    title: str
    ok: bool
    detail: str


def python_syntax() -> Check:
    failures: list[str] = []
    count = 0
    for folder in ("sevir_world", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            count += 1
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as error:
                failures.append(f"{path.relative_to(ROOT)}: {error}")
    return Check("Python 语法", not failures, f"{count} 个文件" if not failures else "; ".join(failures[:5]))


def javascript_syntax() -> Check:
    try:
        result = subprocess.run(
            ["node", "--check", str(ROOT / "web" / "app.js")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check("前端语法", False, str(error))
    return Check("前端语法", result.returncode == 0, (result.stderr or "web/app.js 通过").strip())


def unified_architecture() -> Check:
    retired = (
        "direct.py", "reflective.py", "task_planning.py", "claim_gate.py",
        "guidance.py", "planning.py", "checkpoints.py",
    )
    agent_root = ROOT / "sevir_world" / "agent"
    leftovers = [name for name in retired if (agent_root / name).exists()]
    server = (ROOT / "sevir_world" / "api" / "server.py").read_text(encoding="utf-8")
    ok = not leftovers and "execute_unified_request" in server and "run_reflective_mode" not in server
    return Check("统一 Agent 主链", ok, "唯一执行入口" if ok else "残留：" + ", ".join(leftovers))


def unrestricted_execution() -> Check:
    worker = (ROOT / "sevir_world" / "code_execution" / "worker.py").read_text(encoding="utf-8")
    safety = (ROOT / "sevir_world" / "code_execution" / "safety.py").read_text(encoding="utf-8")
    ok = "exec(compile(" in worker and "namespace.get(\"result\"" in worker and "def run(" not in safety
    return Check("普通 Python 执行", ok, "exec(code, namespace)，run() 仅兼容" if ok else "执行协议仍被限制")


def no_hidden_weather_planner() -> Check:
    app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    visualization = (
        ROOT / "sevir_world" / "tools" / "visualization.py"
    ).read_text(encoding="utf-8")
    forbidden_files = [
        ROOT / "sevir_world" / "tools" / "advisor.py",
        ROOT / "scripts" / "generate_reviewed_answer_figures.py",
        ROOT / "scripts" / "inject_reviewed_answer_figures.py",
    ]
    reasons: list[str] = []
    if any(path.exists() for path in forbidden_files):
        reasons.append("旧规划/图表脚本仍存在")
    if "function buildQuestion()" in app:
        reasons.append("前端仍构造固定问题")
    if "uploaded_ensemble_divergence" in visualization:
        reasons.append("uploaded_ensemble_divergence 预写分析仍存在")
    return Check(
        "无问题级预写分析",
        not reasons,
        "未发现已登记的预写路线" if not reasons else "；".join(reasons),
    )


def api_surface_consistency() -> Check:
    app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    server = (ROOT / "sevir_world" / "api" / "server.py").read_text(encoding="utf-8")
    stale_render_calls = "/api/render-jobs/" in app and "/api/render-jobs/" not in server
    verify_script = (ROOT / "scripts" / "verify_agent_pipeline.py").read_text(encoding="utf-8")
    stale_verifier = "SafeExecutionClient" in verify_script or "agent.artifacts" in verify_script
    reasons: list[str] = []
    if stale_render_calls:
        reasons.append("前端调用服务器未注册的 render-jobs API")
    if stale_verifier:
        reasons.append("verify_agent_pipeline.py 仍引用已删除接口")
    return Check(
        "正式接口一致性",
        not reasons,
        "前端、服务器和验证脚本接口一致" if not reasons else "；".join(reasons),
    )


def refactor_governance() -> Check:
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "refactor_gate.py"), "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        report = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return Check("重构治理契约", False, str(error))
    phases = report.get("phases", [])
    accepted = report.get("accepted_phases", [])
    detail = "；".join(
        f"{item.get('id')}={item.get('status')}({item.get('accepted')}/{item.get('total')})"
        for item in phases
    )
    detail += f"；已验收阶段={','.join(accepted) if accepted else '无'}"
    return Check("重构治理契约", bool(report.get("governance_valid")), detail)


def question_set() -> Check:
    path = ROOT / "data" / "evaluation" / "resident_questions_42.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload["questions"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return Check("42 题验收集", False, str(error))
    numbers = [int(item.get("number", -1)) for item in questions]
    corpus = json.dumps(payload, ensure_ascii=False)
    ok = numbers == list(range(1, 43)) and CASE_ID in corpus and "Aurora" in corpus and "CasCast" in corpus
    return Check("42 题验收集", ok, f"{len(questions)} 题，R190/Aurora/CasCast 覆盖={'是' if ok else '否'}")


def fixed_model_policy() -> Check:
    client = (ROOT / "sevir_world" / "api" / "model_client.py").read_text(encoding="utf-8")
    env_text = (ROOT / ".env").read_text(encoding="utf-8")
    ok = (
        "MODEL_FIXED_NAME=deepseek-v4-flash" in env_text
        and "return [ModelCandidate(name, provider, model, \"fixed\")]" in client
        and "未尝试其他模型" in client
    )
    return Check("固定单模型策略", ok, "deepseek-v4-flash 单次失败即返回" if ok else "固定模型链路不完整")


def main() -> int:
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = [
        python_syntax(), javascript_syntax(), unified_architecture(),
        unrestricted_execution(), no_hidden_weather_planner(),
        question_set(), fixed_model_policy(), api_surface_consistency(),
        refactor_governance(),
    ]
    print("SEVIR Unified Agent 冒烟检查（不能代替阶段验收）")
    print("=" * 48)
    for index, check in enumerate(checks, 1):
        print(f"{index}. [{'通过' if check.ok else '失败'}] {check.title}：{check.detail}")
    failures = [check for check in checks if not check.ok]
    print("-" * 48)
    print(f"结论：{len(checks) - len(failures)}/{len(checks)} 项通过。")
    print("阶段是否完成只以 scripts/refactor_gate.py --require-phase <phase_id> 为准。")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
