"""Download the pinned native Transformers files; never install dependencies."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time

import requests

from .config import MODELS_ROOT
MODELS = {
    "embedding": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", "pytorch_model.bin"),
    "reranker": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "model.safetensors"),
}


def download(item):
    role, (repo, revision, weights) = item
    target = MODELS_ROOT / role
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", weights):
        path = target / name
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            for attempt in range(5):
                try:
                    # A fresh request avoids persisting an expired CDN redirect.
                    with requests.get(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        expected = int(response.headers.get("Content-Length", 0))
                        with partial.open("wb") as stream:
                            for block in response.iter_content(4 * 1024 * 1024):
                                stream.write(block)
                        if expected and partial.stat().st_size != expected:
                            raise ValueError(f"incomplete download: {name}")
                    partial.replace(path)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"file": name, "bytes": path.stat().st_size, "sha256": digest})
        print(role, name, path.stat().st_size, flush=True)
    (target / "provenance.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, MODELS.items()))
