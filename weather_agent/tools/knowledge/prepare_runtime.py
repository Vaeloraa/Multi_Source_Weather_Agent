"""Prepare FP16 safetensors on WSL's native disk, keeping the existing venv intact."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from weather_agent.config import load_dotenv
from weather_agent.tools.knowledge.models import _linux_path, model_root
from weather_agent.tools.gpu import gpu_slot
import os


if __name__ == "__main__":
    load_dotenv()
    python = os.environ["WEATHER_AGENT_RAG_PYTHON"]
    if os.name != "nt" or not python.startswith("/"):
        raise SystemExit("This helper prepares the configured Windows/WSL runtime only")
    job = ROOT / "data/runtime/rag_runtime_setup.json"
    output = job.with_suffix(".result.json")
    job.parent.mkdir(parents=True, exist_ok=True)
    job.write_text(json.dumps({"operation": "prepare", "model_root": _linux_path(model_root()), "output": _linux_path(output)}), encoding="utf-8")
    command = ["wsl.exe", "-d", os.getenv("WEATHER_AGENT_RAG_WSL_DISTRO", "Ubuntu-24.04"), "--exec", python,
               _linux_path(ROOT / "weather_agent/tools/knowledge/model_worker.py"), _linux_path(job)]
    with gpu_slot(timeout=600):
        subprocess.run(command, check=True, timeout=600)
    runtime_root = json.loads(output.read_text(encoding="utf-8"))["model_root"]
    path = ROOT / ".env"
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.startswith("WEATHER_AGENT_RAG_INFERENCE_MODEL_DIR=")]
    path.write_text("\n".join(lines) + "\nWEATHER_AGENT_RAG_INFERENCE_MODEL_DIR=" + runtime_root + "\n", encoding="utf-8")
    print("Inference model files prepared at", runtime_root)
