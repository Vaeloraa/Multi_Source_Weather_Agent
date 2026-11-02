from __future__ import annotations

import argparse
import json
from pathlib import Path

from weather_agent.agent.case_memory import CaseMemoryStore, case_memory_path
from weather_agent.agent.session import SESSION_ROOT, session_database_path
from weather_agent.persistence.sqlite_sessions import SQLiteSessionStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the evidence-only cross-session case memory index."
    )
    parser.add_argument("--session-root", type=Path, default=SESSION_ROOT)
    parser.add_argument("--index", type=Path, default=None)
    args = parser.parse_args()
    store = CaseMemoryStore(args.index or case_memory_path(args.session_root))
    sessions = SQLiteSessionStore(session_database_path(args.session_root))
    turns = (
        (session_id, turn)
        for session_id in sessions.available_ids()
        for turn in sessions.load(session_id)["turns"]
    )
    print(json.dumps(
        {**store.rebuild(turns), **store.status()},
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
