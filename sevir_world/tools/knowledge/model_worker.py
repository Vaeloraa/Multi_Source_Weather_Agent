"""Standalone worker: requires torch 2.7/cu128 and transformers 4.38.2 only.

Run as a file, never importing the host application or its configuration.
"""
import json
import os
import sys
import time
from pathlib import Path
import shutil

PROCESS_STARTED = time.perf_counter()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import transformers

DEPENDENCIES_LOADED = time.perf_counter()

def main():
    started = PROCESS_STARTED
    with open(sys.argv[1], encoding="utf-8") as stream:
        job = json.load(stream)
    if job["operation"] == "prepare":
        destination = Path.home() / ".cache/sevir-rag-models"
        torch.set_num_threads(4)
        for role in ("embedding", "reranker"):
            source = Path(job["model_root"]) / role
            target = destination / role
            provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
            marker = target / "provenance.json"
            if marker.exists() and json.loads(marker.read_text())["revision"] == provenance["revision"]:
                continue
            target.mkdir(parents=True, exist_ok=True)
            cls = AutoModel if role == "embedding" else AutoModelForSequenceClassification
            model = cls.from_pretrained(str(source), local_files_only=True, torch_dtype=torch.float16)
            model.save_pretrained(str(target), safe_serialization=True)
            del model
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"):
                shutil.copyfile(source / name, target / name)
            provenance["runtime_format"] = "fp16 safetensors on WSL native filesystem"
            marker.write_text(json.dumps(provenance), encoding="utf-8")
            print(role, "prepared", flush=True)
        Path(job["output"]).write_text(json.dumps({"model_root": str(destination)}), encoding="utf-8")
        return
    embedding = job["operation"] == "embed"
    path = os.path.join(job["model_root"], "embedding" if embedding else "reranker")
    with open(os.path.join(path, "provenance.json"), encoding="utf-8") as stream:
        provenance = json.load(stream)
    if provenance["revision"] != job["revision"]:
        raise RuntimeError("inference weights differ from the pinned model revision")
    device = job["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RAG requires the configured CUDA device; CPU fallback is not enabled")
    torch.set_num_threads(4)
    loading = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    tokenizer_loaded = time.perf_counter()
    cls = AutoModel if embedding else AutoModelForSequenceClassification
    model = cls.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.eval().to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    model_loaded = time.perf_counter()
    tokenization_seconds = 0.0
    inference_seconds = 0.0
    values = []
    inputs = job["inputs"]
    with torch.inference_mode():
        for start in range(0, len(inputs), 8):
            batch_started = time.perf_counter()
            batch = inputs[start:start + 8]
            if embedding:
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            else:
                encoded = tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch], padding=True,
                                    truncation=True, max_length=768, return_tensors="pt")
            encoded_at = time.perf_counter()
            tokenization_seconds += encoded_at - batch_started
            result = model(**encoded.to(device))
            if embedding:
                result = torch.nn.functional.normalize(result.last_hidden_state[:, 0].float(), p=2, dim=1)
            else:
                result = result.logits.view(-1).float()
            values.append(result.cpu().numpy())
            inference_seconds += time.perf_counter() - encoded_at
            print(f"{job['operation']} {min(start + 8, len(inputs))}/{len(inputs)}", file=sys.stderr, flush=True)
    result = np.concatenate(values) if values else np.empty((0, 1024) if embedding else (0,), dtype=np.float32)
    np.save(job["output"], result.astype(np.float32), allow_pickle=False)
    with open(job["output"] + ".json", "w", encoding="utf-8") as stream:
        json.dump({"torch": torch.__version__, "transformers": transformers.__version__, "device": device,
                   "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                   "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else 0,
                   "operation": job["operation"], "inputs": len(inputs),
                   "timings": {
                       "dependency_import_seconds": round(DEPENDENCIES_LOADED - started, 6),
                       "tokenizer_load_seconds": round(tokenizer_loaded - loading, 6),
                       "model_load_seconds": round(model_loaded - tokenizer_loaded, 6),
                       "tokenization_seconds": round(tokenization_seconds, 6),
                       "inference_seconds": round(inference_seconds, 6),
                   },
                   "worker_seconds": round(time.perf_counter() - started, 3)}, stream)


if __name__ == "__main__":
    main()
