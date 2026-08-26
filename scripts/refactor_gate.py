"""Read-only governance gate for the architecture refactor.

The gate never promotes a criterion or phase.  It validates the persisted
state, checks known counter-examples, and refuses a ``--require-phase`` claim
unless both recorded evidence and automated probes support it.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "architecture_refactor_state.json"
PLAN_PATH = ROOT / "ARCHITECTURE_REFACTOR_PLAN.md"
BACKLOG_PATH = ROOT / "ARCHITECTURE_BACKLOG.md"

CRITERION_STATUSES = {
    "not_started", "implemented", "verified", "accepted", "blocked"
}
PHASE_STATUSES = {"not_started", "in_progress", "verified", "accepted", "blocked"}


@dataclass(frozen=True)
class Probe:
    criterion_id: str
    ok: bool
    detail: str


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _test_contains(marker: str) -> bool:
    return any(
        marker in _read(path)
        for path in (ROOT / "tests").glob("test_*.py")
    )


def _probe_p1_01() -> Probe:
    consumers = (
        ROOT / "sevir_world" / "agent" / "unified.py",
        ROOT / "sevir_world" / "code_execution" / "client.py",
        ROOT / "sevir_world" / "code_execution" / "worker.py",
    )
    connected = any(
        "DomainResultError" in _read(path) or "domain_result_error" in _read(path)
        for path in consumers
    )
    tested = _test_contains("test_domain_result_error_counts_as_execution_failure")
    return Probe(
        "P1-01",
        connected and tested,
        f"业务错误接入主链={'是' if connected else '否'}；专项测试={'是' if tested else '否'}",
    )


def _probe_p1_02() -> Probe:
    tested = _test_contains("test_shared_deadline_bounds_the_complete_request")
    service = _read(ROOT / "sevir_world" / "api" / "unified_service.py")
    old_default = 'SEVIR_PHASE1_AGENT_TIMEOUT_SECONDS", "300"' in service
    return Probe(
        "P1-02",
        tested and not old_default,
        f"300秒旧默认={'仍存在' if old_default else '已移除'}；共享截止时间专项测试={'是' if tested else '否'}",
    )


def _probe_p1_03() -> Probe:
    tested = _test_contains("test_client_cancellation_stops_agent_and_worker")
    return Probe("P1-03", tested, f"取消链专项测试={'是' if tested else '否'}")


def _probe_p1_04() -> Probe:
    tested = _test_contains("test_last_valid_observation_survives_final_model_failure")
    return Probe("P1-04", tested, f"有效Observation保真测试={'是' if tested else '否'}")


def _probe_p1_05() -> Probe:
    preflight = ROOT / "scripts" / "preflight_r190627_demo.py"
    question_set = ROOT / "data" / "evaluation" / "resident_questions_42.json"
    tested = _test_contains("test_r190_representative_trace_preserves_real_statuses")
    ready = preflight.is_file() and question_set.is_file() and tested
    return Probe(
        "P1-05",
        ready,
        (
            "R190预检、代表问题集和真实缓存trace测试存在"
            if ready
            else "R190预检、代表问题集或真实缓存trace测试缺失"
        ),
    )


def _probe_p2_01() -> Probe:
    tested = _test_contains("test_agent_admission") or _test_contains("same request-scoped")
    return Probe("P2-01", tested, f"持久命名空间专项测试={'是' if tested else '否'}")


def _probe_p2_02() -> Probe:
    source_text = "\n".join(
        _read(path)
        for path in (ROOT / "sevir_world").rglob("*.py")
        if ".test_runtime" not in str(path)
    )
    absent = "uploaded_ensemble_divergence" not in source_text
    tested = _test_contains("test_generated_reference_derives_omitted_fields_from_value_classes")
    return Probe(
        "P2-02",
        absent and tested,
        f"问题级预写流程={'已移除' if absent else '仍存在'}；替代路径测试={'是' if tested else '否'}",
    )


def _probe_p2_03() -> Probe:
    tested = _test_contains("test_active_capability_count") and "LazyToolInterface" in _read(
        ROOT / "sevir_world" / "agent" / "runtime.py"
    )
    return Probe("P2-03", tested, f"按需导入专项测试={'是' if tested else '否'}")


def _probe_p2_04() -> Probe:
    app = _read(ROOT / "web" / "app.js")
    stale_ui = "/api/render-jobs/" in app
    stale_modules = any(
        (ROOT / "sevir_world" / "api" / name).exists()
        for name in ("render_jobs.py", "render_jobs_v2.py")
    )
    broken_script = "SafeExecutionClient" in _read(
        ROOT / "scripts" / "verify_agent_pipeline.py"
    )
    tested = _test_contains("test_frontend_only_calls_registered_server_routes")
    ok = not stale_ui and not stale_modules and not broken_script and tested
    return Probe(
        "P2-04",
        ok,
        "；".join((
            f"旧前端render接口={'有' if stale_ui else '无'}",
            f"旧render模块={'有' if stale_modules else '无'}",
            f"失效验证导入={'有' if broken_script else '无'}",
            f"接口一致性测试={'是' if tested else '否'}",
        )),
    )


def _probe_p2_05() -> Probe:
    tested = _test_contains("test_cache_manifest_preserves_source_roles_and_no_answer_evidence")
    return Probe("P2-05", tested, f"统一缓存协议测试={'是' if tested else '否'}")


def _probe_p3_01() -> Probe:
    base = _read(ROOT / "sevir_world" / "capabilities" / "base.py")
    catalog = _read(ROOT / "sevir_world" / "tools" / "sevir" / "Catalog.py")
    tested = _test_contains("test_exact_subject_identity_and_continuous_hour_compose")
    ok = "sevir_subject_key" in base and "_catalog_subject_key" in catalog and tested
    return Probe("P3-01", ok, f"精确SEVIR身份与组合测试={'通过' if ok else '缺失'}")


def _probe_p3_02() -> Probe:
    sevir = _read(ROOT / "sevir_world" / "tools" / "sevir" / "Sevir.py")
    cascast = _read(ROOT / "sevir_world" / "tools" / "cascast" / "CasCast.py")
    ok = all(marker in sevir for marker in ("minutes_before", "minutes_after")) and all(
        marker in cascast for marker in ("uses_future_observations", "subject_kind")
    ) and _test_contains("test_public_time_fields_do_not_alias_legacy_analysis_time")
    return Probe("P3-02", ok, f"时间/缺测/来源公共字段={'完整' if ok else '不完整'}")


def _probe_p3_03() -> Probe:
    tested = all(_test_contains(name) for name in (
        "test_cascast_exposes_identity_missing_values_and_forecast_mode",
        "test_aurora_returns_every_requested_lead_and_availability",
    ))
    return Probe("P3-03", tested, f"四能力组合反证={'是' if tested else '否'}")


def _probe_p3_04() -> Probe:
    tested = all(_test_contains(name) for name in (
        "test_prior_request_facts_do_not_become_new_request_facts",
        "test_promoted_subject_binds_event_and_exact_key_together",
    ))
    return Probe("P3-04", tested, f"请求事实/证据决定分离测试={'是' if tested else '否'}")


def _probe_p3_05() -> Probe:
    worker = _read(ROOT / "sevir_world" / "code_execution" / "worker.py")
    response = _read(ROOT / "sevir_world" / "api" / "response_pipeline.py")
    tested = _test_contains("test_failed_worker_step_does_not_commit_partial_variables")
    ok = "working_namespace" in worker and 'result.pop("progress_steps", None)' in response and tested
    return Probe("P3-05", ok, f"事务式Worker与单时间线={'是' if ok else '否'}")


def _probe_p3_06() -> Probe:
    ensemble = _read(ROOT / "sevir_world" / "data_access" / "ensemble.py")
    config = _read(ROOT / "sevir_world" / "data_access" / "config.py")
    manifests = "\n".join(_read(path) for path in (ROOT / "data" / "foundation_live_cache" / "aurora").glob("*/manifest.json"))
    ok = (
        "CASCAST_EVENT_ID" not in config
        and 'requested in {"sample", "event_001", ""}' not in ensemble
        and "legacy_case_cache_alias" not in manifests
        and "R19062708588472" not in manifests
    )
    return Probe("P3-06", ok, f"运行时默认事件/区域缓存别名={'无' if ok else '仍存在'}")


def _probe_p3_07() -> Probe:
    tested = _test_contains("class Phase3PublicContractTests")
    fixed = "never retry with another model" in _read(ROOT / "sevir_world" / "api" / "model_client.py")
    return Probe("P3-07", tested and fixed, f"新契约与固定模型适配器={'存在' if tested and fixed else '缺失'}")


def _probe_p3_08() -> Probe:
    acceptance_roots = list((ROOT / "data" / "evaluation").glob("acceptance-42-phase3-*"))
    valid_runs: list[str] = []
    rejected: list[str] = []
    for root in acceptance_roots:
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            results = manifest.get("results") or []
            complete = (
                int(manifest.get("question_count", 0)) == 42
                and int(manifest.get("result_count", 0)) == 42
                and not bool(manifest.get("stopped_early", False))
                and len(results) == 42
                and {int(item.get("number", 0)) for item in results} == set(range(1, 43))
                and all(item.get("status") == "passed" for item in results)
                and all(float(item.get("elapsed_seconds", 999)) <= 90 for item in results)
                and all(bool(str(item.get("answer") or "").strip()) for item in results)
                and all(bool(item.get("artifacts")) for item in results)
            )
        except (OSError, ValueError, TypeError):
            complete = False
        (valid_runs if complete else rejected).append(root.name)
    detail = (
        f"完整42/42验收={len(valid_runs)}；不完整/冒烟记录={len(rejected)}"
    )
    return Probe("P3-08", bool(valid_runs), detail)


PROBES: dict[str, Callable[[], Probe]] = {
    "P1-01": _probe_p1_01,
    "P1-02": _probe_p1_02,
    "P1-03": _probe_p1_03,
    "P1-04": _probe_p1_04,
    "P1-05": _probe_p1_05,
    "P2-01": _probe_p2_01,
    "P2-02": _probe_p2_02,
    "P2-03": _probe_p2_03,
    "P2-04": _probe_p2_04,
    "P2-05": _probe_p2_05,
    "P3-01": _probe_p3_01,
    "P3-02": _probe_p3_02,
    "P3-03": _probe_p3_03,
    "P3-04": _probe_p3_04,
    "P3-05": _probe_p3_05,
    "P3-06": _probe_p3_06,
    "P3-07": _probe_p3_07,
    "P3-08": _probe_p3_08,
}

BASELINE_ACCEPTANCE = {
    "B-01": {"P2-01"},
    "B-02": {"P2-02"},
    "B-03": {"P2-04"},
    "B-04": {"P1-01"},
    "B-05": {"P1-02", "P1-03"},
    "B-06": {"P3-01"},
    "B-07": {"P3-02"},
    "B-08": {"P3-03"},
    "B-09": {"P3-03"},
    "B-10": {"P3-06"},
}


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_state(state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if state.get("schema_version") != 1:
        errors.append("schema_version 必须为 1")
    phases = state.get("phases")
    if not isinstance(phases, list) or not phases:
        return errors + ["phases 必须是非空列表"]
    phase_ids = [str(item.get("id", "")) for item in phases]
    if len(phase_ids) != len(set(phase_ids)) or any(not item for item in phase_ids):
        errors.append("phase id 必须非空且唯一")
    active_phase = str(state.get("active_phase", ""))
    if active_phase not in phase_ids:
        errors.append("active_phase 不存在")
    active_statuses = 0
    seen_criterion_ids: set[str] = set()
    accepted_criterion_ids: set[str] = set()
    previous_accepted = True
    for phase in phases:
        phase_id = str(phase.get("id", ""))
        phase_status = str(phase.get("status", ""))
        if phase_status not in PHASE_STATUSES:
            errors.append(f"{phase_id}: 非法阶段状态 {phase_status}")
        if phase_status in {"in_progress", "verified"}:
            active_statuses += 1
        if phase_id == active_phase and phase_status not in {
            "in_progress", "verified", "blocked", "accepted"
        }:
            errors.append(f"{phase_id}: active_phase 不能处于 {phase_status}")
        if not phase.get("allowed_paths"):
            errors.append(f"{phase_id}: 缺少范围锁定 allowed_paths")
        if not previous_accepted and phase_status not in {"not_started", "blocked"}:
            errors.append(f"{phase_id}: 前一阶段未验收，不能进入 {phase_status}")
        acceptance = phase.get("acceptance")
        if not isinstance(acceptance, list) or not acceptance:
            errors.append(f"{phase_id}: acceptance 必须是非空列表")
            previous_accepted = phase_status == "accepted"
            continue
        criterion_statuses: list[str] = []
        for criterion in acceptance:
            criterion_id = str(criterion.get("id", ""))
            status = str(criterion.get("status", ""))
            criterion_statuses.append(status)
            if not criterion_id or criterion_id in seen_criterion_ids:
                errors.append(f"{phase_id}: 验收项 id 必须非空且全局唯一")
            seen_criterion_ids.add(criterion_id)
            if status not in CRITERION_STATUSES:
                errors.append(f"{criterion_id}: 非法状态 {status}")
                continue
            required = set(criterion.get("required_evidence") or [])
            evidence = criterion.get("evidence") or []
            if not required:
                errors.append(f"{criterion_id}: required_evidence 不能为空")
            for index, item in enumerate(evidence, 1):
                if not isinstance(item, dict):
                    errors.append(f"{criterion_id}: 第 {index} 条证据不是对象")
                    continue
                missing_fields = [
                    field for field in (
                        "type", "reference", "result", "recorded_at", "notes"
                    )
                    if not str(item.get(field, "")).strip()
                ]
                if missing_fields:
                    errors.append(
                        f"{criterion_id}: 第 {index} 条证据缺少 "
                        + ", ".join(missing_fields)
                    )
                if item.get("result") not in {"pass", "fail"}:
                    errors.append(f"{criterion_id}: 证据 result 只能是 pass/fail")
            passed_types = {
                str(item.get("type", ""))
                for item in evidence
                if isinstance(item, dict) and item.get("result") == "pass"
            }
            if status in {"verified", "accepted"}:
                missing = sorted(required - passed_types)
                if missing:
                    errors.append(f"{criterion_id}: 缺少通过证据 {', '.join(missing)}")
            if status == "accepted":
                accepted_criterion_ids.add(criterion_id)
                probe = PROBES.get(criterion_id)
                if probe is None:
                    errors.append(f"{criterion_id}: 没有自动反证检查")
                else:
                    result = probe()
                    if not result.ok:
                        errors.append(f"{criterion_id}: 自动反证未通过：{result.detail}")
        if phase_status == "accepted" and any(
            status != "accepted" for status in criterion_statuses
        ):
            errors.append(f"{phase_id}: 阶段 accepted 但仍有未 accepted 验收项")
        if phase_status == "verified" and any(
            status not in {"verified", "accepted"} for status in criterion_statuses
        ):
            errors.append(f"{phase_id}: 阶段 verified 但仍有未验证验收项")
        previous_accepted = phase_status == "accepted"
    if active_statuses > 1:
        errors.append("最多只能有一个 in_progress/verified 阶段")
    baseline = state.get("known_baseline")
    if not isinstance(baseline, list):
        errors.append("known_baseline 必须是列表")
    else:
        baseline_ids = {str(item.get("id", "")) for item in baseline}
        missing_baselines = sorted(set(BASELINE_ACCEPTANCE) - baseline_ids)
        if missing_baselines:
            errors.append("缺少反证基线：" + ", ".join(missing_baselines))
        for item in baseline:
            baseline_id = str(item.get("id", ""))
            requirements = BASELINE_ACCEPTANCE.get(baseline_id)
            if requirements is None:
                continue
            resolved = item.get("resolved") is True
            criteria_accepted = requirements <= accepted_criterion_ids
            if resolved and not criteria_accepted:
                errors.append(
                    f"{baseline_id}: 标为已解决，但验收项未全部 accepted"
                )
            if criteria_accepted and not resolved:
                errors.append(
                    f"{baseline_id}: 验收项已 accepted，但反证基线仍未更新"
                )
    if not PLAN_PATH.is_file():
        errors.append("缺少 ARCHITECTURE_REFACTOR_PLAN.md")
    if not BACKLOG_PATH.is_file():
        errors.append("缺少 ARCHITECTURE_BACKLOG.md")
    plan_text = _read(PLAN_PATH)
    for criterion_id in seen_criterion_ids:
        if criterion_id not in plan_text:
            errors.append(f"{criterion_id}: 人类可读计划中缺少该验收项")
    return errors


def collect_report(state: dict[str, Any]) -> dict[str, Any]:
    errors = validate_state(state)
    probes = [probe() for probe in PROBES.values()]
    phases = []
    for phase in state.get("phases", []):
        acceptance = phase.get("acceptance") or []
        phases.append({
            "id": phase.get("id"),
            "title": phase.get("title"),
            "status": phase.get("status"),
            "accepted": sum(item.get("status") == "accepted" for item in acceptance),
            "total": len(acceptance),
        })
    return {
        "governance_valid": not errors,
        "errors": errors,
        "active_phase": state.get("active_phase"),
        "phases": phases,
        "probes": [asdict(item) for item in probes],
        "accepted_phases": [
            item["id"] for item in phases if item["status"] == "accepted"
        ],
    }


def phase_claim_errors(
    state: dict[str, Any], report: dict[str, Any], phase_id: str
) -> list[str]:
    errors = list(report["errors"])
    phase = next(
        (item for item in state.get("phases", []) if item.get("id") == phase_id),
        None,
    )
    if phase is None:
        return errors + [f"未知阶段：{phase_id}"]
    if phase.get("status") != "accepted":
        errors.append(f"{phase_id} 当前状态为 {phase.get('status')}，不是 accepted")
    criterion_ids = {item.get("id") for item in phase.get("acceptance", [])}
    for probe in report["probes"]:
        if probe["criterion_id"] in criterion_ids and not probe["ok"]:
            errors.append(
                f"{probe['criterion_id']} 自动反证未通过：{probe['detail']}"
            )
    return errors


def _print_report(report: dict[str, Any]) -> None:
    print("架构重构治理验收门（只读，不自动提升状态）")
    print("=" * 56)
    print(f"治理文件：{'有效' if report['governance_valid'] else '无效'}")
    print(f"当前阶段：{report['active_phase']}")
    for phase in report["phases"]:
        print(
            f"- {phase['id']} | {phase['status']} | "
            f"accepted {phase['accepted']}/{phase['total']} | {phase['title']}"
        )
    print("自动反证：")
    for probe in report["probes"]:
        marker = "通过" if probe["ok"] else "未通过"
        print(f"- [{marker}] {probe['criterion_id']}：{probe['detail']}")
    if report["errors"]:
        print("状态错误：")
        for error in report["errors"]:
            print(f"- {error}")
    if report["accepted_phases"]:
        print("已验收阶段：" + ", ".join(report["accepted_phases"]))
    else:
        print("已验收阶段：无。不得宣称任何重构阶段已经完成。")


def main() -> int:
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--require-phase")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        state = load_state(args.state)
    except (OSError, ValueError) as error:
        print(f"无法读取治理状态：{error}", file=sys.stderr)
        return 2
    report = collect_report(state)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    if args.require_phase:
        claim_errors = phase_claim_errors(
            state, report, str(args.require_phase)
        )
        if claim_errors:
            if not args.json:
                print("阶段验收：失败")
                for error in claim_errors:
                    print(f"- {error}")
            return 1
        if not args.json:
            print(f"阶段验收：通过（{args.require_phase}）")
    return 0 if report["governance_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
