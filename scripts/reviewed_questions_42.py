from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QUESTION_SET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evaluation"
    / "resident_questions_42.json"
)


def load_question_set() -> dict[str, Any]:
    return json.loads(QUESTION_SET_PATH.read_text(encoding="utf-8"))


def question_tuples() -> dict[int, tuple[str, str]]:
    payload = load_question_set()
    return {
        int(item["number"]): (str(item["payload_kind"]), str(item["prompt"]))
        for item in payload["questions"]
    }


def follow_up_chain() -> list[str]:
    payload = load_question_set()
    return [str(value) for value in payload["follow_up_chain_q30"]]

