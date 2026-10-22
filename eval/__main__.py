"""Run the evaluation suite:  python -m eval [--no-cache] [--thinking low|medium|high]

Sections are added as the pipeline grows (extraction now; verifier and latency/cost later).
"""
import argparse

from eval import extraction
from src import llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="ignore cached responses (fresh timing run)")
    parser.add_argument("--thinking", choices=["low", "medium", "high"], default=None)
    args = parser.parse_args()
    llm.set_cache(not args.no_cache)
    extraction.run(thinking_level=args.thinking)


if __name__ == "__main__":
    main()
