"""Paths for knowledge sources, model files and the local retrieval index."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = KNOWLEDGE_ROOT / "sources.json"
ORIGINALS_ROOT = KNOWLEDGE_ROOT / "originals"
MODELS_ROOT = KNOWLEDGE_ROOT / "models"
INDEX_PATH = KNOWLEDGE_ROOT / "index.sqlite3"
