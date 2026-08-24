"""Validate one ERA5–Aurora case before allowing Agent interpretation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sevir_world.data_access.foundation import get_case_foundation_artifacts
from sevir_world.data_access.foundation_validation import build_foundation_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_id", nargs="?", default="R19080121067677")
    args = parser.parse_args()
    case = ROOT / "data" / "era5_cases" / args.event_id / "artifacts"
    evidence_path = case / "case_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else None
    registry = get_case_foundation_artifacts(args.event_id)
    bundle = build_foundation_bundle(args.event_id, evidence, registry.get("artifacts", []))
    result = {
        "event_id": args.event_id,
        "status": bundle["status"],
        "validation": bundle["validation"],
        "required_leads": [6, 12, 18, 24],
        "available_artifacts": [x["name"] for x in registry.get("artifacts", []) if x.get("available")],
    }
    out = case / "foundation_case_validation.json"
    try:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output"] = str(out)
    except PermissionError:
        result["output"] = "stdout_only_permission_denied"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
