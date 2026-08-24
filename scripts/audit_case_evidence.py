"""Validate that a case has the evidence and boundaries required for reporting."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    event_id = sys.argv[1] if len(sys.argv) > 1 else "R19080121067677"
    case_dir = ROOT / "data" / "era5_cases" / event_id
    evidence_path = case_dir / "artifacts" / "case_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    required = {"sevir", "cascast", "era5", "aurora"}
    present = set(evidence) & required
    missing = sorted(required - present)
    artifact_names = {p.name for p in (case_dir / "artifacts").iterdir()}
    verification = evidence.get("aurora", {}).get("verification_artifacts", [])
    missing_artifacts = sorted(set(verification) - artifact_names)
    result = {
        "event_id": event_id,
        "evidence_file": str(evidence_path.relative_to(ROOT)),
        "sources_present": sorted(present),
        "missing_sources": missing,
        "aurora_verification_status": evidence.get("aurora", {}).get("verification_status"),
        "missing_verification_artifacts": missing_artifacts,
        "pass": not missing and not missing_artifacts and evidence.get("aurora", {}).get("verification_status") == "full_domain_multi_lead",
    }
    out = ROOT / "data" / f"{event_id}_evidence_audit.json"
    try:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output"] = str(out.relative_to(ROOT))
    except PermissionError:
        # Read-only/locked deployments still get the complete audit on stdout.
        result["output"] = "stdout_only_permission_denied"
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
