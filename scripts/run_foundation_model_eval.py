"""Run the foundation-model question set against a temporary local server."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parents[1]
PORT = 8014


def post(case: dict, session_id: str) -> dict:
    payload_data = {
        "question": case["question"],
        "data_source": "cascast",
        "provider": "qwen",
        "session_id": session_id,
    }
    # Make abnormal-case fixtures explicit in the request payload.  A natural
    # language question mentioning a conflict/unavailability is not itself
    # evidence that the API should treat the input as conflicting/missing.
    if case.get("id") == "id_conflict":
        payload_data["uploaded_event_id"] = "R19050812577422"
    return_payload = json.dumps(payload_data).encode("utf-8")
    request = Request(
        f"http://127.0.0.1:{PORT}/api/run",
        data=return_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    cases = json.loads((ROOT / "data" / "foundation_model_eval_cases.json").read_text(encoding="utf-8"))["cases"]
    requested_ids = {item.strip() for item in __import__("os").environ.get("FOUNDATION_EVAL_CASE_IDS", "").split(",") if item.strip()}
    if requested_ids:
        cases = [case for case in cases if case.get("id") in requested_ids]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT)
    env["SEVIR_DEMO_PORT"] = str(PORT)
    server = subprocess.Popen(
        [sys.executable, "-m", "sevir_world.api.server"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    rows = []
    result_root = ROOT / "data" / "foundation_round3"
    try:
        result_root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        result_root = Path(tempfile.gettempdir()) / "sevir_foundation_round3"
        result_root.mkdir(parents=True, exist_ok=True)
    try:
        time.sleep(3)
        def run_one(case: dict) -> dict:
            started = time.perf_counter()
            row = {"id": case["id"], "question": case["question"]}
            try:
                result = post(case, f"foundation-eval-{case['id']}")
                row.update({
                    "status": result.get("status", "ok"),
                    "selected_tools": result.get("selected_tools", []),
                    "tool_calls": result.get("tool_calls", []),
                    "route_mode": result.get("route_mode"),
                    "agent_warning": result.get("agent_warning"),
                    "answer": result.get("answer", ""),
                    "request_id": result.get("request_id"),
                    "trace_file": result.get("trace_file"),
                    "request_trace": result.get("request_trace"),
                    "trace": result.get("trace"),
                })
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                row.update({"status": "request_failed", "error": str(error)})
            row["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            return row

        # Keep model-service concurrency bounded; ERA5 file reads are cheap,
        # but twelve simultaneous LLM turns cause avoidable timeouts.
        with ThreadPoolExecutor(max_workers=min(3, len(cases))) as pool:
            futures = [pool.submit(run_one, case) for case in cases]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                try:
                    (result_root / f"{row['id']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                except PermissionError:
                    pass
                print(json.dumps(row, ensure_ascii=False)[:1200], flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    output = result_root / "summary.json"
    try:
        output.write_text(json.dumps({"cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    except PermissionError:
        output = ROOT / "data" / "era5_cases" / "R19080121067677" / "artifacts" / "foundation_model_agent_eval.json"
        output.write_text(json.dumps({"cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
