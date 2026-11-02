"""Explicitly import legacy JSON sessions into SQLite and index their evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sevir_world.agent.case_memory import CaseMemoryStore, case_memory_path
from sevir_world.agent.session import SESSION_ROOT, normalize_session_id, session_database_path
from sevir_world.persistence.sqlite_sessions import SQLiteSessionStore


def migrate_sessions(root: Path) -> dict[str, int]:
    store = SQLiteSessionStore(session_database_path(root))
    memory = CaseMemoryStore(case_memory_path(root))
    imported = skipped = indexed = 0
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("_"):
            continue
        session_id = normalize_session_id(path.stem)
        state = json.loads(path.read_text(encoding="utf-8"))
        if store.import_session(session_id, state, updated_at=path.stat().st_mtime_ns):
            imported += 1
        else:
            skipped += 1
        for turn in store.load(session_id)["turns"]:
            if memory.upsert_turn(session_id, turn) is not None:
                indexed += 1
    return {"imported_sessions": imported, "existing_sessions": skipped,
            "indexed_evidence_records": indexed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=SESSION_ROOT)
    args = parser.parse_args()
    print(json.dumps(migrate_sessions(args.session_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
