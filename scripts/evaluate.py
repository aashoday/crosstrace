"""Run ranking systems on a split.

Example:
    uv run python scripts/evaluate.py --candidates data/candidates.jsonl \
        --queries data/queries.jsonl --split time
"""

from __future__ import annotations

import argparse

from crosstrace.benchmark import leave_one_project_out, time_split
from crosstrace.evaluation import (
    SYSTEMS,
    evaluate_split,
    format_table,
    load_candidates,
    load_queries,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--split", choices=["time", "lopo"], default="time")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--out", default="results/eval.json")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    queries = load_queries(args.queries)
    if args.split == "time":
        splits = [time_split(queries)]
    else:
        splits = list(leave_one_project_out(queries).values())

    results = []
    for split in splits:
        result = evaluate_split(split, candidates, args.systems)
        results.append(result)
        print(format_table(result))
        print()
    meta = {
        "split": args.split,
        "systems": args.systems,
        "candidates_file": args.candidates,
        "queries_file": args.queries,
        "n_candidates": len(candidates),
        "n_queries": len(queries),
    }
    save_results(results, args.out, meta)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()