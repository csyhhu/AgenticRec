"""Public dataset benchmark runner.

Usage:
    # Run with default MovieLens-1M (download required)
    python -m agentic_rec.bench_public

    # Specify dataset path
    python -m agentic_rec.bench_public --data-dir ~/ml-1m

    # Run with JSON output
    python -m agentic_rec.bench_public --json

    # Use a smaller subset for quick testing
    python -m agentic_rec.bench_public --max-tasks 30
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .bench import (
    RunRow,
    Scenario,
    evaluate_agentic,
    evaluate_baseline,
    hot_baseline,
    item_knn_baseline,
    print_report,
    summarize,
    tag_baseline,
)
from .datasets import MovieLensAdapter


def run_public_benchmark(
    data_dir: Optional[str] = None,
    top_k: int = 5,
    max_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full benchmark on MovieLens-1M.

    Parameters
    ----------
    data_dir : str | None
        Path to the MovieLens-1M data directory.
    top_k : int
        Number of items to recommend per task.
    max_tasks : int | None
        If set, cap the number of tasks per scenario (for quick testing).

    Returns
    -------
    report : dict
        Structured benchmark report (same format as bench.run_benchmark).
    """
    print("Loading MovieLens-1M dataset...")
    adapter = MovieLensAdapter(raw_path=data_dir)
    corpus = adapter.load_corpus()
    scenarios = adapter.load_scenarios()

    # Cap tasks if requested
    if max_tasks is not None:
        for scenario in scenarios:
            scenario.tasks = scenario.tasks[:max_tasks]

    total_tasks = sum(len(s.tasks) for s in scenarios)
    print(f"Corpus: {len(corpus)} items, Scenarios: {len(scenarios)}, Tasks: {total_tasks}\n")

    # Pre-compute neighbor profiles for collaboration
    print("Building neighbor profiles...")
    neighbor_profiles = adapter.build_neighbor_profiles()
    print(f"Neighbor profiles: {len(neighbor_profiles)}")

    # Build user_train_map for ItemKNN (from MovieLens raw ratings, training split only)
    print("Building user-train map for ItemKNN...")
    user_train_map = adapter.build_user_train_map(max_users=500)
    print(f"User-train map: {len(user_train_map)} users\n")

    # Build a helper to inject neighbor profiles into the pipeline
    def agentic_runner_with_profiles(corpus, top_k, **kwargs):
        from .pipeline import AgenticPipeline
        from .bench import BenchTask

        def run(task: BenchTask) -> tuple:
            pipe = AgenticPipeline(
                corpus=corpus,
                top_n=top_k,
                neighbor_profiles=neighbor_profiles,
                **kwargs,
            )
            if task.profile_tags:
                pipe.memory.update_profile(task.user_id, tags=task.profile_tags)
            result = pipe.run(task.query, user_id=task.user_id, scene=task.scene)
            # trace_cost inline
            cost = float(len(result.trace) + 2 * sum(
                1 for ev in result.trace if ev.get("action") == "veto"
            ))
            return result.items, result.total_ms, len(result.trace), cost

        return run

    # ---- Methods ----
    print("Evaluating methods...")
    methods: Dict[str, List[RunRow]] = {}

    # AgenticRec-Gated
    print("  [1/6] AgenticRec-Gated ...")
    rows_gated: List[RunRow] = []
    runner = agentic_runner_with_profiles(
        corpus, top_k, enable_collaboration=True, adaptive_collaboration=True
    )
    for scenario in scenarios:
        for task in scenario.tasks:
            items, latency, steps, cost = runner(task)
            rows_gated.append(RunRow(
                scenario=scenario.name, user_id=task.user_id, query=task.query,
                recommended=[it.id for it in items], relevant=task.relevant_ids,
                latency_ms=latency, trace_steps=steps, trace_cost=cost,
            ))
    methods["AgenticRec-Gated"] = rows_gated

    # AgenticRec-Collab
    print("  [2/6] AgenticRec-Collab ...")
    rows_collab: List[RunRow] = []
    runner = agentic_runner_with_profiles(
        corpus, top_k, enable_collaboration=True, adaptive_collaboration=False
    )
    for scenario in scenarios:
        for task in scenario.tasks:
            items, latency, steps, cost = runner(task)
            rows_collab.append(RunRow(
                scenario=scenario.name, user_id=task.user_id, query=task.query,
                recommended=[it.id for it in items], relevant=task.relevant_ids,
                latency_ms=latency, trace_steps=steps, trace_cost=cost,
            ))
    methods["AgenticRec-Collab"] = rows_collab

    # AgenticRec-Core
    print("  [3/6] AgenticRec-Core ...")
    rows_core: List[RunRow] = []
    runner = agentic_runner_with_profiles(
        corpus, top_k, enable_collaboration=False
    )
    for scenario in scenarios:
        for task in scenario.tasks:
            items, latency, steps, cost = runner(task)
            rows_core.append(RunRow(
                scenario=scenario.name, user_id=task.user_id, query=task.query,
                recommended=[it.id for it in items], relevant=task.relevant_ids,
                latency_ms=latency, trace_steps=steps, trace_cost=cost,
            ))
    methods["AgenticRec-Core"] = rows_core

    # ItemKNN
    print("  [4/6] ItemKNN ...")
    methods["ItemKNN"] = evaluate_baseline(
        "ItemKNN", item_knn_baseline(corpus, user_train_map, top_k), scenarios, top_k
    )

    # TagBaseline
    print("  [5/6] TagBaseline ...")
    methods["TagBaseline"] = evaluate_baseline(
        "TagBaseline", tag_baseline(corpus, top_k), scenarios, top_k
    )

    # HotBaseline
    print("  [6/6] HotBaseline ...")
    methods["HotBaseline"] = evaluate_baseline(
        "HotBaseline", hot_baseline(corpus, top_k), scenarios, top_k
    )

    # ---- Summarise ----
    return {
        "top_k": top_k,
        "corpus_size": len(corpus),
        "tasks": total_tasks,
        "methods": {
            name: {
                "summary": summarize(rows, corpus, top_k),
                "rows": [row.__dict__ for row in rows],
            }
            for name, rows in methods.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AgenticRec-Bench on MovieLens-1M."
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to MovieLens-1M data directory (default: ~/.agentic_rec/ml-1m/)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Cap tasks per scenario (for quick testing)")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON instead of table")
    args = parser.parse_args()

    try:
        report = run_public_benchmark(
            data_dir=args.data_dir,
            top_k=args.top_k,
            max_tasks=args.max_tasks,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nDownload MovieLens-1M from:", file=sys.stderr)
        print("  https://grouplens.org/datasets/movielens/1m/", file=sys.stderr)
        print(f"  Extract to ~/.agentic_rec/ml-1m/ (or pass --data-dir)", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
