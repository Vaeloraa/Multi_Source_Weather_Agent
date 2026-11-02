"""Safe local citation pages; PDF links refer to the exact indexed artifact."""
import hashlib
from html import escape
import json

from .config import ORIGINALS_ROOT
from .store import ROOT, KnowledgeStore


def source_response(chunk_id, *, original=False):
    store = KnowledgeStore()
    record = store.read(chunk_id)
    with store.connect() as connection:
        row = connection.execute("SELECT metadata FROM documents WHERE doc_id=? AND revision=?",
                                 (record["doc_id"], record["revision"])).fetchone()
    metadata = json.loads(row[0])
    path = (ROOT / metadata["local_file"]).resolve()
    if not path.is_relative_to(ORIGINALS_ROOT.resolve()):
        raise ValueError("invalid original source path")
    if original:
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != metadata["sha256"]:
            raise ValueError("original source checksum changed")
        # Never serve source HTML/notebook scripts as an executable web page.
        return ("application/pdf" if metadata["format"] == "pdf" else "text/plain; charset=utf-8"), body
    page = record["locator"].get("page")
    source_link = f"{record['url']}/source" + (f"#page={page}" if page else "")
    text = ("<!doctype html><html lang='zh'><meta charset='utf-8'><title>资料原文</title>"
            "<style>body{max-width:960px;margin:40px auto;padding:20px;font:17px/1.6 sans-serif}pre{white-space:pre-wrap}</style>"
            f"<h1>{escape(record['title'])}</h1><p>{escape(record['version'])} · 修订 {record['revision']}</p>"
            f"<p>{escape(record['section'])}</p><p>位置：{escape(json.dumps(record['locator'], ensure_ascii=False))}</p>"
            f"<p><a href='{escape(source_link, quote=True)}'>本地原始资料</a> · "
            f"<a href='{escape(record['source_url'], quote=True)}' rel='noreferrer'>来源网站</a></p>"
            f"<pre>{escape(record['text'])}</pre></html>")
    return "text/html; charset=utf-8", text.encode("utf-8")
