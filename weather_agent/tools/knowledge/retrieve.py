"""Dense + BM25 recall, reciprocal rank fusion and local cross-encoder reranking."""
import os
import time
from copy import deepcopy

import numpy as np

from .ingest import embedding_text
from .models import EMBEDDING_REVISION, PIPELINE_VERSION, RERANKER_REVISION, LocalModels
from .store import KnowledgeStore


class Retriever:
    def __init__(self, store=None, models=None):
        self.store = store or KnowledgeStore()
        self.models = models or LocalModels()

    def candidates(self, query, vector, tools=None):
        identity = self.store.describe()["identity"]
        if identity.get("embedding_revision") != EMBEDDING_REVISION or identity.get("pipeline") != PIPELINE_VERSION:
            raise RuntimeError("RAG index model/pipeline differs; rebuild the index before querying")
        return self.store.candidates(query, vector, tools)

    @staticmethod
    def select(candidates, scores, limit=6):
        threshold = float(os.getenv("WEATHER_AGENT_RAG_MIN_RERANK", "0.05"))
        result = []
        for item, logit in sorted(zip(candidates, scores), key=lambda pair: (-float(pair[1]), pair[0]["chunk_id"])):
            score = float(1 / (1 + np.exp(-np.clip(logit, -80, 80))))
            if score < threshold:
                continue
            # Overlapping windows should not consume the complete context budget.
            words = set(item["text"].split())
            if any(item["doc_id"] == old["doc_id"] and len(words & set(old["text"].split())) / max(1, len(words | set(old["text"].split()))) > .8 for old in result):
                continue
            result.append({**item, "rerank_score": score, "rank": len(result) + 1})
            if len(result) >= limit:
                break
        return result

    def search(self, query, tools=None, limit=6):
        query = str(query).strip()
        if not query or len(query) > 4000:
            raise ValueError("query must contain 1 to 4000 characters")
        if tools is not None and (not isinstance(tools, list) or any(t not in {"sevir", "cascast", "era5", "aurora"} for t in tools)):
            raise ValueError("tools must be a list drawn from sevir, cascast, era5, aurora")
        if not 1 <= int(limit) <= 10:
            raise ValueError("limit must be between 1 and 10")
        start = time.perf_counter()
        self.store.describe()  # Fail before model loading when no index exists.
        timings = {"index_check_seconds": round(time.perf_counter() - start, 6)}
        model_runs = {}
        if tools == []:
            candidates = []
        else:
            stage = time.perf_counter()
            vector = self.models.embed([query])[0]
            timings["embedding_seconds"] = round(time.perf_counter() - stage, 6)
            if isinstance(getattr(self.models, "last_run", None), dict):
                model_runs["embedding"] = deepcopy(self.models.last_run)
            stage = time.perf_counter()
            candidates = self.candidates(query, vector, tools)
            timings["recall_seconds"] = round(time.perf_counter() - stage, 6)
        stage = time.perf_counter()
        scores = self.models.rerank([[query, embedding_text(item)] for item in candidates]) if candidates else []
        timings["rerank_seconds"] = round(time.perf_counter() - stage, 6)
        if candidates and isinstance(getattr(self.models, "last_run", None), dict):
            model_runs["reranker"] = deepcopy(self.models.last_run)
        stage = time.perf_counter()
        records = self.select(candidates, scores, limit)
        timings["selection_seconds"] = round(time.perf_counter() - stage, 6)
        return {"status": "ok" if records else "partial", "query": query, "records": records,
                "record_count": len(records), "candidate_count": len(candidates),
                "reranker_revision": RERANKER_REVISION,
                "min_rerank_score": float(os.getenv("WEATHER_AGENT_RAG_MIN_RERANK", "0.05")),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "timings": timings, "model_runs": model_runs,
                "score_notice": "retrieval relevance scores, never truth probabilities"}
