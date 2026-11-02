"""Parse only manifest-approved originals and publish complete document indexes."""
import hashlib
import json
from pathlib import Path
import re

from .models import EMBEDDING_REVISION, PIPELINE_VERSION, LocalModels, tokenizer
from .config import MANIFEST_PATH, ORIGINALS_ROOT
from .store import ROOT, KnowledgeStore


def extract(path, kind):
    if kind == "pdf":
        from pypdf import PdfReader
        segments = [{"page": i, "text": page.extract_text() or ""} for i, page in enumerate(PdfReader(path).pages, 1)]
        if any(len(item["text"].strip()) < 30 for item in segments):
            raise ValueError("PDF has empty/scanned pages; supply a text PDF or an explicitly reviewed OCR extraction")
        return segments
    if kind == "ipynb":
        document = json.loads(path.read_text(encoding="utf-8"))
        return [{"cell": i, "cell_type": cell["cell_type"], "text": "".join(cell.get("source", []))}
                for i, cell in enumerate(document["cells"], 1) if cell["cell_type"] in {"markdown", "code"}]
    if kind == "html":
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            element.decompose()
        body = soup.select_one("#main-content") or soup.find("main") or soup.find("article") or soup.body or soup
        lines = [line.strip() for line in body.get_text("\n").splitlines() if line.strip()]
        text = "\n".join(lines)
    elif kind in {"md", "txt"}:
        text = path.read_text(encoding="utf-8-sig")
    else:
        raise ValueError(f"unsupported source format: {kind}")
    return [{"line_start": 1, "line_end": len(text.splitlines()), "text": text}]


def split_chunks(source, segments, encoder):
    """Keep page/cell boundaries and Markdown model-family heading context."""
    output = []
    heading_stack = []
    for segment in segments:
        text = segment["text"]
        inherited_section = " / ".join(h for _, h in heading_stack)
        boundaries = [(0, "")]
        if source["format"] in {"md", "ipynb"}:
            # Fenced code is source material, never executable ingestion input.
            fenced = False
            offset = 0
            for line in text.splitlines(keepends=True):
                if line.lstrip().startswith("```"):
                    fenced = not fenced
                match = re.match(r"^(#{1,6})\s+(.+)", line) if not fenced else None
                if match:
                    depth = len(match[1])
                    heading_stack = [(d, h) for d, h in heading_stack if d < depth]
                    heading_stack.append((depth, match[2].strip()))
                    if depth <= 2:
                        boundaries.append((offset, " / ".join(h for _, h in heading_stack)))
                offset += len(line)
        boundaries.append((len(text), ""))
        section = inherited_section if source["format"] == "ipynb" else ""
        for (begin, heading), (end, _) in zip(boundaries, boundaries[1:]):
            section = heading or section
            part = text[begin:end]
            encoding = encoder.encode(part, add_special_tokens=False)
            offsets = encoding.offsets
            position = 0
            while position < len(offsets):
                stop = min(position + 448, len(offsets))
                char_start, char_end = offsets[position][0], offsets[stop - 1][1]
                # Prefer a paragraph boundary in the last quarter of the window.
                if stop < len(offsets):
                    floor = offsets[min(position + 336, stop - 1)][0]
                    paragraph = part.rfind("\n\n", floor, char_end)
                    if paragraph > floor:
                        while stop > position + 1 and offsets[stop - 1][1] > paragraph:
                            stop -= 1
                        char_end = offsets[stop - 1][1]
                snippet = part[char_start:char_end].strip()
                if len(snippet) > 20:
                    locator = {k: v for k, v in segment.items() if k != "text"}
                    if "line_start" in locator:
                        locator["line_start"] = segment["line_start"] + text.count("\n", 0, begin + char_start)
                        locator["line_end"] = segment["line_start"] + text.count("\n", 0, begin + char_end)
                    locator.update(char_start=begin + char_start, char_end=begin + char_end)
                    output.append({"doc_id": source["doc_id"], "revision": source["revision"], "title": source["title"],
                                   "section": section, "text": snippet, "locator": locator})
                if stop == len(offsets):
                    break
                position = max(position + 1, stop - 64)
    return output


def ingest(manifest=None, *, store=None, models=None):
    path = Path(manifest or MANIFEST_PATH)
    document = json.loads(path.read_text(encoding="utf-8"))
    store = store or KnowledgeStore()
    sources = []
    all_chunks = []
    encoder = tokenizer()
    seen = set()
    for item in document["sources"]:
        if item.get("enabled", True) is False:
            continue
        if item["doc_id"] in seen or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", item["doc_id"]):
            raise ValueError("doc_id must be unique and contain only lower-case letters, digits and hyphens")
        seen.add(item["doc_id"])
        source_path = (ROOT / item["local_file"]).resolve()
        if not source_path.is_relative_to(ORIGINALS_ROOT.resolve()):
            raise ValueError("only approved knowledge/originals files can be ingested")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"source hash differs from manifest: {item['doc_id']}")
        source = {key: item[key] for key in ("doc_id", "title", "tools", "format", "version", "source_url", "local_file", "revision", "sha256")}
        for key in ("artifact_source_url", "downloaded_version"):
            if key in item:
                source[key] = item[key]
        if int(source["revision"]) < 1:
            raise ValueError("source revision must be positive")
        segments = extract(source_path, source["format"])
        if sum(len(s["text"].strip()) for s in segments) < 100:
            raise ValueError("source has insufficient text: " + source["doc_id"])
        all_chunks.extend(split_chunks(source, segments, encoder))
        sources.append(source)
    identity = {"pipeline": PIPELINE_VERSION, "embedding_revision": EMBEDDING_REVISION,
                "sources": [{k: s[k] for k in ("doc_id", "revision", "sha256")} for s in sources]}
    identity["manifest_digest"] = hashlib.sha256(json.dumps(sources, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if store.path.exists() and store.describe()["identity"] == identity:
        return {**store.describe(), "changed": False}
    inputs = [embedding_text(chunk) for chunk in all_chunks]
    vectors = (models or LocalModels()).embed(inputs)
    store.publish(sources, all_chunks, vectors, identity)
    return {**store.describe(), "changed": True}


def embedding_text(chunk):
    # Keep the full passage within the native encoder's 512-token window.
    prefix = chunk["section"] + "\n" + chunk["title"]
    offsets = tokenizer().encode(prefix, add_special_tokens=False).offsets
    if len(offsets) > 40:
        prefix = prefix[:offsets[39][1]]
    return prefix + "\n" + chunk["text"]
