"""MovieLens-1M full benchmark script.

Runs all methods (AgenticRec variants + baselines including ItemKNN) on
MovieLens-1M and prints a formatted comparison table.

Usage:
    # Quick test (30 tasks per scenario)
    python examples/run_movielens_bench.py --max-tasks 30

    # Full run
    python examples/run_movielens_bench.py

    # Specify data directory
    python examples/run_movielens_bench.py --data-dir D:/ml-1m

    # Export to JSON
    python examples/run_movielens_bench.py --json --output report.json

    # Custom top-K
    python examples/run_movielens_bench.py --top-k 10 --max-tasks 50
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_rec.bench import (
    RunRow,
    evaluate_agentic,
    evaluate_baseline,
    hot_baseline,
    item_knn_baseline,
    summarize,
    tag_baseline,
)
from agentic_rec.datasets import MovieLensAdapter


# ---------------------------------------------------------------------------
# Rich table printer (no external dependency)
# ---------------------------------------------------------------------------

def print_results_table(
    report: Dict[str, Any],
    top_k: int,
    total_tasks: int,
    corpus_size: int,
    elapsed_sec: float,
) -> None:
    """Print a well-formatted comparison table."""

    metric_names = [
        f"HR@{top_k}", f"Recall@{top_k}", f"MRR@{top_k}", f"NDCG@{top_k}",
        "Coverage", "Diversity", "Latency(ms)", "Steps", "Cost",
    ]
    key_map = [
        f"hit_rate@{top_k}", f"recall@{top_k}", f"mrr@{top_k}", f"ndcg@{top_k}",
        "coverage", "diversity", "latency_ms", "trace_steps", "trace_cost",
    ]

    def fmt(v: float, is_pct: bool = True) -> str:
        if is_pct:
            return f"{v * 100:.2f}%"
        return f"{v:.2f}"

    pct_indices = {0, 1, 2, 3, 4, 5}  # HR/Recall/MRR/NDCG/Coverage/Diversity are percentages

    methods_data = report["methods"]
    method_names = list(methods_data.keys())

    # ---- Overall Table ----
    print("=" * 100)
    print(f"  MovieLens-1M Benchmark  |  tasks={total_tasks}  corpus={corpus_size}  top_k={top_k}  time={elapsed_sec:.1f}s")
    print("=" * 100)
    print()

    # Header
    col_widths = [max(22, max(len(n) for n in method_names) + 2)]
    col_widths += [max(10, len(m)) + 2 for m in metric_names]
    header = "Method".ljust(col_widths[0])
    for i, m in enumerate(metric_names):
        header += m.rjust(col_widths[i + 1])
    print(header)
    print("-" * sum(col_widths))

    # Rows
    for method in method_names:
        overall = methods_data[method]["summary"]["overall"]
        row = method.ljust(col_widths[0])
        for i, key in enumerate(key_map):
            val = overall.get(key, 0)
            is_pct = i in pct_indices
            row += fmt(val, is_pct).rjust(col_widths[i + 1])
        print(row)

    # ---- By Scenario ----
    print()
    print("-" * 100)
    print("  By Scenario (HR / Recall / MRR / NDCG)")
    print("-" * 100)

    scenario_names = list(methods_data[method_names[0]]["summary"]["by_scenario"].keys())
    scenario_keys = [f"hit_rate@{top_k}", f"recall@{top_k}", f"mrr@{top_k}", f"ndcg@{top_k}"]

    for scenario in scenario_names:
        print(f"\n  [{scenario}]")
        s_header = "Method".ljust(col_widths[0])
        for key in scenario_keys:
            s_header += key.replace(f"@{top_k}", "").rjust(10)
        print(s_header)
        print("  " + "-" * (col_widths[0] + 40))

        for method in method_names:
            metrics = methods_data[method]["summary"]["by_scenario"].get(scenario, {})
            row = ("  " + method).ljust(col_widths[0])
            for key in scenario_keys:
                val = metrics.get(key, 0)
                row += f"{val * 100:.2f}%".rjust(10)
            print(row)

    # ---- Summary Stats ----
    print()
    print("-" * 100)
    print("  Additional Statistics")
    print("-" * 100)

    # GateRate and VetoRate for AgenticRec variants
    for method in method_names:
        if not method.startswith("AgenticRec"):
            continue
        rows = methods_data[method]["rows"]
        gate_enabled = sum(1 for r in rows if r.get("trace_steps", 0) > 4)
        veto_count = 0
        for r in rows:
            # Veto is inferred: trace_cost > trace_steps means at least one veto
            if r.get("trace_cost", 0) > r.get("trace_steps", 0):
                veto_count += 1
        gate_rate = gate_enabled / max(1, len(rows))
        veto_rate = veto_count / max(1, len(rows))
        print(f"  {method}: GateRate={gate_rate:.1%}  VetoRate={veto_rate:.1%}  AvgSteps={sum(r.get('trace_steps',0) for r in rows)/max(1,len(rows)):.1f}")

    print()
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full MovieLens-1M benchmark with ItemKNN baseline."
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to MovieLens-1M data directory (default: ~/.agentic_rec/ml-1m/)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K for evaluation (default: 5)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Cap tasks per scenario for quick testing (e.g. 30)")
    parser.add_argument("--json", action="store_true",
                        help="Export results as JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: stdout)")
    args = parser.parse_args()

    t_start = time.time()

    # 1. Load data
    print("=" * 60)
    print("  Loading MovieLens-1M dataset...")
    adapter = MovieLensAdapter(raw_path=args.data_dir)
    corpus = adapter.load_corpus()
    scenarios = adapter.load_scenarios()

    if args.max_tasks is not None:
        for scenario in scenarios:
            scenario.tasks = scenario.tasks[:args.max_tasks]

    total_tasks = sum(len(s.tasks) for s in scenarios)
    print(f"  Corpus: {len(corpus)} items")
    for s in scenarios:
        print(f"  {s.name}: {len(s.tasks)} tasks")
    print(f"  Total: {total_tasks} tasks")

    # 2. Build shared resources
    print("\n  Building neighbor profiles...")
    neighbor_profiles = adapter.build_neighbor_profiles()
    print(f"  Profiles: {len(neighbor_profiles)}")

    print("  Building user-train map for ItemKNN...")
    user_train_map = adapter.build_user_train_map(max_users=500)
    print(f"  Users with training data: {len(user_train_map)}")

    # 3. Helper: agentic runner with neighbor profiles
    def make_agentic_runner(corpus, top_k, **kwargs):
        from agentic_rec.pipeline import AgenticPipeline

        def run(task):
            pipe = AgenticPipeline(
                corpus=corpus, top_n=top_k,
                neighbor_profiles=neighbor_profiles, **kwargs,
            )
            if task.profile_tags:
                pipe.memory.update_profile(task.user_id, tags=task.profile_tags)
            result = pipe.run(task.query, user_id=task.user_id, scene=task.scene)
            cost = float(len(result.trace) + 2 * sum(
                1 for ev in result.trace if ev.get("action") == "veto"
            ))
            return result.items, result.total_ms, len(result.trace), cost
        return run

    # 4. Run all methods
    print("\n" + "=" * 60)
    print("  Evaluating methods...")
    methods: Dict[str, List[RunRow]] = {}

    # AgenticRec-Gated
    print("  [1/6] AgenticRec-Gated ...", end=" ", flush=True)
    t0 = time.time()
    rows = []
    runner = make_agentic_runner(corpus, args.top_k,
                                 enable_collaboration=True, adaptive_collaboration=True)
    for scenario in scenarios:
        for task in scenario.tasks:
            items, lat, steps, cost = runner(task)
            rows.append(RunRow(
                scenario=scenario.name, user_id=task.user_id, query=task.query,
                recommended=[it.id for it in items], relevant=task.relevant_ids,
                latency_ms=lat, trace_steps=steps, trace_cost=cost,
            ))
    methods["AgenticRec-Gated"] = rows
    print(f"done ({time.time() - t0:.1f}s)")

    # AgenticRec-Collab
    print("  [2/6] AgenticRec-Collab ...", end=" ", flush=True)
    t0 = time.time()
    rows = []
    runner = make_agentic_runner(corpus, args.top_k,
                                 enable_collaboration=True, adaptive_collaboration=False)
    for scenario in scenarios:
        for task in scenario.tasks:
            items, lat, steps, cost = runner(task)
            rows.append(RunRow(
                scenario=scenario.name, user_id=task.user_id, query=task.query,
                recommended=[it.id for it in items], relevant=task.relevant_ids,
                latency_ms=lat, trace_steps=steps, trace_cost=cost,
            ))
    methods["AgenticRec-Collab"] = rows
    print(f"done ({time.time() - t0:.1f}s)")

    # AgenticRec-Core
    print("  [3/6] AgenticRec-Core ...", end=" ", flush=True)
    t0 = time.time()
    rows = []
    runner = make_agentic_runner(corpus, args.top_k, enable_collaboration=False)
    for scenario in scenarios:
        for task in scenario.tasks:
            items, lat, steps, cost = runner(task)
            rows.append(RunRow(
                scenario=scenario.name, user_id=task.user_id, query=task.query,
                recommended=[it.id for it in items], relevant=task.relevant_ids,
                latency_ms=lat, trace_steps=steps, trace_cost=cost,
            ))
    methods["AgenticRec-Core"] = rows
    print(f"done ({time.time() - t0:.1f}s)")

    # ItemKNN
    print("  [4/6] ItemKNN ...", end=" ", flush=True)
    t0 = time.time()
    methods["ItemKNN"] = evaluate_baseline(
        "ItemKNN", item_knn_baseline(corpus, user_train_map, args.top_k),
        scenarios, args.top_k,
    )
    print(f"done ({time.time() - t0:.1f}s)")

    # TagBaseline
    print("  [5/6] TagBaseline ...", end=" ", flush=True)
    t0 = time.time()
    methods["TagBaseline"] = evaluate_baseline(
        "TagBaseline", tag_baseline(corpus, args.top_k), scenarios, args.top_k,
    )
    print(f"done ({time.time() - t0:.1f}s)")

    # HotBaseline
    print("  [6/6] HotBaseline ...", end=" ", flush=True)
    t0 = time.time()
    methods["HotBaseline"] = evaluate_baseline(
        "HotBaseline", hot_baseline(corpus, args.top_k), scenarios, args.top_k,
    )
    print(f"done ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t_start

    # 5. Build report
    report = {
        "top_k": args.top_k,
        "corpus_size": len(corpus),
        "tasks": total_tasks,
        "methods": {
            name: {
                "summary": summarize(rows, corpus, args.top_k),
                "rows": [row.__dict__ for row in rows],
            }
            for name, rows in methods.items()
        },
    }

    # 6. Output
    output = []
    if args.json:
        output.append(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        # Capture table to string
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        print_results_table(report, args.top_k, total_tasks, len(corpus), elapsed)
        output.append(sys.stdout.getvalue())
        sys.stdout = old_stdout

    result_text = "\n".join(output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"Results saved to {args.output}")
    else:
        print(result_text)


if __name__ == "__main__":
    main()
