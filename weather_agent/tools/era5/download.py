"""Prefetch large CDS selections outside the interactive Agent's deadline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weather_agent.config import load_dotenv
from weather_agent.tools.era5.cds import retrieve_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Exact CDS dataset ID")
    parser.add_argument("--selection", required=True, type=Path, help="JSON file containing the native CDS/MARS request")
    parser.add_argument("--split-by", choices=("month", "day", "none"), default="month")
    parser.add_argument("--cached", action="store_true", help="Verify/use cache without remote downloads")
    args = parser.parse_args()
    load_dotenv()
    try:
        selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
        result = retrieve_data(args.dataset, selection, split_by=args.split_by,
                               data_policy="cached" if args.cached else "live")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
