"""Resolve source markers exclusively against real, recorded tool evidence."""
import re


def resolve_citations(answer, calls):
    evidence = {int(item["chunk_id"]): item for call in calls
                if call.get("api") in {"knowledge.search", "knowledge.read"}
                for item in call.get("knowledge_evidence", [])}
    used = {}
    invalid = []

    def replace(match):
        chunk_id = int(match[1])
        item = evidence.get(chunk_id)
        if item is None:
            invalid.append(chunk_id)
            return "[Unverified source citation]"
        used[chunk_id] = {k: item[k] for k in ("chunk_id", "doc_id", "revision", "title", "version", "locator", "source_url", "url")}
        locator = item["locator"]
        position = f"Page {locator['page']}" if "page" in locator else f"Cell {locator['cell']}" if "cell" in locator else f"Lines {locator.get('line_start', '?')}–{locator.get('line_end', '?')}"
        return f"[{item['doc_id']}, {position}]({item['url']})"

    answer = re.sub(r"\[[^\]]*\]\(/api/knowledge/chunks/(\d+)\)", lambda m: f"[K{m[1]}]", answer)
    answer = re.sub(r"\[K(\d+)\]", replace, answer)
    return answer, list(used.values()), sorted(set(invalid))
