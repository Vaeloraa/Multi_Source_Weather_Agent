"""SQLite document revisions, immutable source chunks and an FTS5 index."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3

import numpy as np

from .config import INDEX_PATH, PROJECT_ROOT as ROOT


def lexical_text(text):
    """Unicode words plus Chinese bigrams, shared by indexing and BM25 queries."""
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    words += [a + b for a, b in zip(words, words[1:]) if len(a) == len(b) == 1 and a >= "\u4e00" and b >= "\u4e00"]
    return " ".join(words)


class KnowledgeStore:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("SEVIR_RAG_INDEX", str(INDEX_PATH)))

    @contextmanager
    def connect(self, *, write=False):
        if not write and not self.path.is_file():
            raise RuntimeError("RAG index is not built; run python -m sevir_world.tools.knowledge.cli ingest")
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if write:
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id TEXT NOT NULL, revision INTEGER NOT NULL, metadata TEXT NOT NULL,
                        PRIMARY KEY (doc_id, revision));
                    CREATE TABLE IF NOT EXISTS active_documents (
                        doc_id TEXT PRIMARY KEY, revision INTEGER NOT NULL);
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL, revision INTEGER NOT NULL,
                        text TEXT NOT NULL, section TEXT NOT NULL, locator TEXT NOT NULL, vector BLOB NOT NULL,
                        FOREIGN KEY(doc_id, revision) REFERENCES documents(doc_id, revision));
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(title, body);
                    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS active_chunks (chunk_id INTEGER PRIMARY KEY REFERENCES chunks(chunk_id));
                """)
            yield connection
            if write:
                connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def publish(self, documents, chunks, vectors, identity):
        """Atomically activate a complete manifest; retain old chunks for citations."""
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count differs")
        if vectors.ndim != 2 or vectors.shape[1] != 1024 or not np.isfinite(vectors).all():
            raise ValueError("invalid BGE vectors")
        with self.connect(write=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM active_chunks")
            # Historical passages stay readable, but cannot change active BM25 IDF.
            connection.execute("DELETE FROM chunk_fts")
            for document in documents:
                existing = connection.execute("SELECT metadata FROM documents WHERE doc_id=? AND revision=?",
                                              (document["doc_id"], document["revision"])).fetchone()
                serialized = json.dumps(document, ensure_ascii=False, sort_keys=True)
                if existing and existing[0] != serialized:
                    raise ValueError("a changed source needs a new manifest revision: " + document["doc_id"])
                connection.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?)",
                                   (document["doc_id"], document["revision"], serialized))
            for chunk, vector in zip(chunks, vectors):
                cursor = connection.execute("INSERT INTO chunks(doc_id,revision,text,section,locator,vector) VALUES(?,?,?,?,?,?)",
                    (chunk["doc_id"], chunk["revision"], chunk["text"], chunk["section"], json.dumps(chunk["locator"]),
                     np.asarray(vector, dtype="<f4").tobytes()))
                connection.execute("INSERT INTO chunk_fts(rowid,title,body) VALUES(?,?,?)",
                    (cursor.lastrowid, lexical_text(chunk["title"] + " " + chunk["section"]), lexical_text(chunk["text"])))
                connection.execute("INSERT INTO active_chunks VALUES(?)", (cursor.lastrowid,))
            connection.execute("DELETE FROM active_documents")
            connection.executemany("INSERT INTO active_documents VALUES(?,?)", [(d["doc_id"], d["revision"]) for d in documents])
            connection.execute("INSERT OR REPLACE INTO settings VALUES('identity',?)", (json.dumps(identity, sort_keys=True),))

    def describe(self):
        with self.connect() as connection:
            identity = json.loads(connection.execute("SELECT value FROM settings WHERE key='identity'").fetchone()[0])
            docs = [json.loads(row[0]) for row in connection.execute("SELECT d.metadata FROM documents d JOIN active_documents a USING(doc_id,revision)")]
            count = connection.execute("SELECT count(*) FROM active_chunks").fetchone()[0]
        return {"status": "ok", "document_count": len(docs), "chunk_count": count, "identity": identity,
                "documents": [{**{k: d[k] for k in ("doc_id", "revision", "title", "tools")},
                               "version": d.get("downloaded_version", d["version"])} for d in docs]}

    @staticmethod
    def record(row):
        metadata = json.loads(row["metadata"])
        locator = json.loads(row["locator"])
        return {"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "revision": row["revision"],
                "title": metadata["title"], "version": metadata.get("downloaded_version", metadata["version"]), "tools": metadata["tools"],
                "text": row["text"], "section": row["section"], "locator": locator,
                "source_url": metadata.get("artifact_source_url", metadata["source_url"]),
                "citation": f"[K{row['chunk_id']}]", "url": f"/api/knowledge/chunks/{row['chunk_id']}"}

    def read(self, chunk_id):
        with self.connect() as connection:
            row = connection.execute("SELECT c.*,d.metadata FROM chunks c JOIN documents d USING(doc_id,revision) WHERE chunk_id=?",
                                     (int(chunk_id),)).fetchone()
        if row is None:
            raise KeyError(f"unknown knowledge chunk: {chunk_id}")
        return self.record(row)

    def candidates(self, query, vector, tools=None, *, dense_limit=30, lexical_limit=20):
        allowed = set(tools or [])
        with self.connect() as connection:
            rows = list(connection.execute("SELECT c.*,d.metadata FROM chunks c JOIN active_chunks a USING(chunk_id) JOIN documents d USING(doc_id,revision)"))
            if tools is not None:
                rows = [row for row in rows if allowed.intersection(json.loads(row["metadata"])["tools"])]
            if not rows:
                return []
            matrix = np.stack([np.frombuffer(row["vector"], dtype="<f4") for row in rows])
            dense = matrix @ np.asarray(vector, dtype=np.float32)
            dense_indices = np.argsort(-dense, kind="stable")[:dense_limit]
            query_words = list(dict.fromkeys(lexical_text(query).split()))[:80]
            lexical = []
            if query_words:
                match = " OR ".join('"' + word + '"' for word in query_words)
                # Apply the allow-list before ranking, including inactive revisions.
                valid = {row["chunk_id"] for row in rows}
                lexical = [(row[0], row[1]) for row in connection.execute(
                    "SELECT rowid,bm25(chunk_fts,2.0,1.0) AS score FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY score,rowid", (match,))
                    if row[0] in valid][:lexical_limit]
        by_id = {row["chunk_id"]: i for i, row in enumerate(rows)}
        scores = {}
        for rank, i in enumerate(dense_indices, 1):
            scores[int(i)] = {"rrf_score": 1 / (60 + rank), "dense_score": float(dense[i])}
        for rank, (chunk_id, bm25) in enumerate(lexical, 1):
            i = by_id[chunk_id]
            score = scores.setdefault(i, {"rrf_score": 0., "dense_score": float(dense[i])})
            score["rrf_score"] += 1 / (60 + rank)
            score["bm25_score"] = float(bm25)
        result = []
        counts = {}
        for i, score in sorted(scores.items(), key=lambda pair: (-pair[1]["rrf_score"], rows[pair[0]]["chunk_id"])):
            doc_id = rows[i]["doc_id"]
            if counts.get(doc_id, 0) >= 6:
                continue
            result.append({**self.record(rows[i]), **score})
            counts[doc_id] = counts.get(doc_id, 0) + 1
            if len(result) == 20:
                break
        return result
