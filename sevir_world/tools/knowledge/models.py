"""Native local BGE inference, isolated from the Agent's Python dependencies."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from functools import lru_cache

import numpy as np

from ..gpu import gpu_slot
from .config import MODELS_ROOT, PROJECT_ROOT as ROOT

EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
PIPELINE_VERSION = "bge-hybrid-v2-heading2-token448-overlap64"


def model_root():
    return Path(os.getenv("SEVIR_RAG_MODEL_DIR", str(MODELS_ROOT))).resolve()


@lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer
    path = model_root() / "embedding/tokenizer.json"
    if not path.is_file():
        raise RuntimeError("RAG models are missing; run python -m sevir_world.tools.knowledge.download_models")
    value = Tokenizer.from_file(str(path))
    value.no_truncation()
    value.no_padding()
    return value


def _linux_path(path):
    path = Path(path).resolve()
    if path.drive:
        return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]
    return str(path)


class LocalModels:
    """Only this short-lived subprocess owns GPU weights; exit frees CUDA memory."""

    def run(self, operation, inputs, *, timeout=300):
        started = time.perf_counter()
        python = os.getenv("SEVIR_RAG_PYTHON", "").strip()
        if not python:
            raise RuntimeError("configure SEVIR_RAG_PYTHON in the project .env")
        role = "embedding" if operation == "embed" else "reranker"
        provenance = json.loads((model_root() / role / "provenance.json").read_text(encoding="utf-8"))
        expected_revision = EMBEDDING_REVISION if role == "embedding" else RERANKER_REVISION
        if provenance["revision"] != expected_revision or any(
                (model_root() / role / item["file"]).stat().st_size != item["bytes"] for item in provenance["files"]):
            raise RuntimeError("local RAG model version/files differ; run python -m sevir_world.tools.knowledge.download_models")
        validated = time.perf_counter()
        use_wsl = os.name == "nt" and python.startswith("/")
        root = ROOT / "data/runtime/rag_jobs"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="job-", dir=root) as directory:
            job = Path(directory)
            translate = _linux_path if use_wsl else lambda path: str(Path(path).resolve())
            runtime_root = os.getenv("SEVIR_RAG_INFERENCE_MODEL_DIR", "").strip()
            payload = {"operation": operation, "inputs": inputs, "model_root": runtime_root or translate(model_root()),
                       "output": translate(job / "output.npy"), "device": os.getenv("SEVIR_RAG_DEVICE", "cuda"),
                       "revision": expected_revision}
            request = job / "input.json"
            request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            command = [python, translate(Path(__file__).with_name("model_worker.py")), translate(request)]
            if use_wsl:
                command = ["wsl.exe", "-d", os.getenv("SEVIR_RAG_WSL_DISTRO", "Ubuntu-24.04"), "--exec",
                           "/usr/bin/timeout", "--kill-after=10", str(timeout), *command]
            waiting = time.perf_counter()
            with gpu_slot(timeout=timeout):
                acquired = time.perf_counter()
                completed = subprocess.run(command, capture_output=True, timeout=timeout + 15)
                exited = time.perf_counter()
            if completed.returncode or not (job / "output.npy").exists():
                detail = completed.stderr.decode("utf-8", errors="replace")[-3000:]
                raise RuntimeError(f"local RAG {operation} failed: {detail}")
            values = np.load(job / "output.npy", allow_pickle=False)
            metadata = job / "output.npy.json"
            self.last_run = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
            self.last_run["host_timings"] = {
                "validation_seconds": round(validated - started, 6),
                "job_setup_seconds": round(waiting - validated, 6),
                "gpu_queue_seconds": round(acquired - waiting, 6),
                "process_seconds": round(exited - acquired, 6),
                "output_read_seconds": round(time.perf_counter() - exited, 6),
            }
            if isinstance(self.last_run.get("worker_seconds"), (int, float)):
                self.last_run["host_timings"]["process_startup_and_exit_seconds"] = round(
                    max(0.0, exited - acquired - self.last_run["worker_seconds"]), 6)
            if not np.isfinite(values).all():
                raise ValueError("model returned non-finite values")
            return values

    def embed(self, texts):
        return self.run("embed", list(texts), timeout=1800 if len(texts) > 16 else 180)

    def rerank(self, pairs):
        return self.run("rerank", list(pairs), timeout=600 if len(pairs) > 40 else 180)
