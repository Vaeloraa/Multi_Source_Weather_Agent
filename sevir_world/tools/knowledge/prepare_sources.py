"""Download an explicit public-source manifest and check extractable evidence.

This prepares documents for RAG development. It does not create an embedding
index, run notebook code, crawl links, or call the Agent/model APIs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request


from .config import KNOWLEDGE_ROOT as KNOWLEDGE, PROJECT_ROOT as ROOT
MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract(path: Path, kind: str) -> list[dict]:
    if kind == "pdf":
        from pypdf import PdfReader

        return [
            {"page": number, "text": page.extract_text() or ""}
            for number, page in enumerate(PdfReader(path).pages, 1)
        ]
    if kind == "ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return [
            {"cell": number, "cell_type": cell["cell_type"], "text": "".join(cell.get("source", []))}
            for number, cell in enumerate(notebook["cells"], 1)
            if cell.get("cell_type") in {"markdown", "code"}
        ]
    if kind == "html":
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            element.decompose()
        body = soup.select_one("#main-content") or soup.find("main") or soup.find("article") or soup.body or soup
        lines = [line.strip() for line in body.get_text("\n").splitlines() if line.strip()]
        return [{"line_start": 1, "line_end": len(lines), "text": "\n".join(lines)}]
    text = path.read_text(encoding="utf-8-sig")
    return [{"line_start": 1, "line_end": len(text.splitlines()), "text": text}]


def fetch(source: dict, destination: Path) -> dict:
    attempts = [{"url": source["download_url"], "version": source["version"]}]
    attempts.extend(source.get("fallback_downloads", []))
    errors = []
    for candidate in attempts:
        try:
            request = urllib.request.Request(candidate["url"], headers={"User-Agent": "SEVIR-Tool-Knowledge-Research/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read(MAX_DOWNLOAD_BYTES + 1)
                resolved_url = response.url
                content_type = response.headers.get("Content-Type", "")
            if len(payload) > MAX_DOWNLOAD_BYTES:
                raise ValueError("document exceeds the 30 MiB download limit")
            if source["format"] == "pdf" and not payload.lstrip().startswith(b"%PDF-"):
                raise ValueError("response is not a PDF")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return {"downloaded_from": resolved_url, "downloaded_version": candidate["version"],
                    "artifact_source_url": candidate.get("source_url", source["source_url"]),
                    "content_type": content_type, "download_errors": errors}
        except Exception as error:
            errors.append({"url": candidate["url"], "error": str(error)})
    raise RuntimeError(json.dumps(errors, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing documents from the explicit manifest")
    args = parser.parse_args()
    manifest_path = KNOWLEDGE / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [item["doc_id"] for item in manifest["sources"]]
    if len(ids) != len(set(ids)):
        raise ValueError("source doc_id values must be unique")
    results = []
    for source in manifest["sources"]:
        doc_id = source["doc_id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", doc_id):
            raise ValueError(f"invalid source doc_id: {doc_id}")
        revision = int(source.get("revision", 1))
        path = KNOWLEDGE / "originals" / doc_id / str(revision) / f"source.{source['format']}"
        if not path.resolve().is_relative_to(KNOWLEDGE.resolve()):
            raise ValueError("source path leaves the knowledge workspace")
        result = {"doc_id": doc_id, "tools": source["tools"], "title": source["title"]}
        try:
            if not path.is_file():
                if not args.download:
                    raise FileNotFoundError("original not downloaded; run with --download")
                source.update(fetch(source, path))
            payload = path.read_bytes()
            segments = extract(path, source["format"])
            characters = sum(len(item["text"].strip()) for item in segments)
            if characters < 500:
                raise ValueError(f"only {characters} extractable characters; inspect for an error page or scan")
            if source["format"] == "html":
                lowered = " ".join(item["text"] for item in segments).lower()
                if "enable javascript and cookies to continue" in lowered or "access denied" in lowered[:200]:
                    raise ValueError("download is an access challenge, not source content")
            save_json(KNOWLEDGE / "extracted" / f"{doc_id}.json", {
                "doc_id": doc_id, "source_url": source.get("artifact_source_url", source["source_url"]),
                "version": source.get("downloaded_version", source["version"]),
                "extraction": "pypdf/basic HTML or source-cell extraction; not a production RAG parser",
                "segments": segments,
            })
            source.update({"local_file": path.relative_to(ROOT).as_posix(), "revision": revision,
                           "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                           "acquisition_status": "downloaded_and_text_checked"})
            source.pop("index_status", None)
            result.update(status="ok", bytes=len(payload), segments=len(segments), text_characters=characters)
            if source["format"] == "pdf":
                result["pages"] = len(segments)
                result["pages_without_text"] = [item["page"] for item in segments if not item["text"].strip()]
        except Exception as error:
            result.update(status="error", error=str(error))
            source["acquisition_status"] = "failed"
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    manifest["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(manifest_path, manifest)
    report = {"scope": manifest["scope"], "source_count": len(results),
              "downloaded_and_text_checked": sum(item["status"] == "ok" for item in results),
              "scope": "Source acquisition and text extraction only; query knowledge.describe() for current index state.",
              "sources": results}
    save_json(KNOWLEDGE / "source_validation.json", report)
    return 0 if all(item["status"] == "ok" for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
