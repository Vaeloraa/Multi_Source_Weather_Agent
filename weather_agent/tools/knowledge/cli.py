"""Ingest, query and describe the approved local document corpus."""
import argparse
import json

from .ingest import ingest
from .retrieve import Retriever
from .store import KnowledgeStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["ingest", "search", "describe"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--tools", nargs="+")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    from ...config import load_dotenv
    load_dotenv()
    if args.action == "ingest":
        result = ingest(args.manifest)
    elif args.action == "search":
        result = Retriever().search(args.query or "", args.tools)
    else:
        result = KnowledgeStore().describe()
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
