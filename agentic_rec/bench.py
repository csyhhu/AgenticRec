"""AgenticRec-Bench: tiny but complete evaluation loop.

The goal is not to replace industrial offline evaluation. It gives the project
an executable credibility layer: scenarios, metrics, baselines, traces, and a
CLI that anyone can run without data downloads or API keys.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .core import Item
from .pipeline import AgenticPipeline


@dataclass
class BenchTask:
    user_id: str
    query: str
    scene: str
    relevant_ids: List[str]
    profile_tags: Dict[str, float] = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    description: str
    tasks: List[BenchTask]


@dataclass
class RunRow:
    scenario: str
    user_id: str
    query: str
    recommended: List[str]
    relevant: List[str]
    latency_ms: float
    trace_steps: int
    trace_cost: float


# ---------------------------------------------------------------------------
# Dataset: intentionally small, semantically readable, and deterministic.
# ---------------------------------------------------------------------------
def default_corpus() -> List[Dict[str, Any]]:
    return [
        {"id": "mv_8821", "title": "雾港谜局", "tags": ["悬疑", "都市"], "author": "A1", "ctr_prior": 0.18, "freshness": 0.9, "hot": 980},
        {"id": "mv_7102", "title": "山雨欲来", "tags": ["悬疑", "犯罪"], "author": "A2", "ctr_prior": 0.15, "freshness": 0.4, "hot": 420},
        {"id": "mv_6010", "title": "轻松小镇日记", "tags": ["治愈", "轻松"], "author": "A3", "ctr_prior": 0.10, "freshness": 0.7, "hot": 220},
        {"id": "mv_5511", "title": "暗夜推理者", "tags": ["悬疑", "推理"], "author": "A1", "ctr_prior": 0.12, "freshness": 0.2, "hot": 1200},
        {"id": "mv_4321", "title": "夜行列车", "tags": ["悬疑", "惊悚"], "author": "A4", "ctr_prior": 0.09, "freshness": 0.6, "hot": 90},
        {"id": "mv_3010", "title": "午后茶馆", "tags": ["治愈"], "author": "A5", "ctr_prior": 0.06, "freshness": 0.5, "hot": 60},
        {"id": "mv_2008", "title": "搞笑同事录", "tags": ["喜剧", "轻松"], "author": "A6", "ctr_prior": 0.20, "freshness": 0.8, "hot": 310},
        {"id": "mv_1207", "title": "迷雾追凶", "tags": ["悬疑", "犯罪"], "author": "A7", "ctr_prior": 0.13, "freshness": 0.3, "hot": 770},
        {"id": "mv_0904", "title": "巷口便利店", "tags": ["治愈", "轻松"], "author": "A8", "ctr_prior": 0.08, "freshness": 0.9, "hot": 130},
        {"id": "mv_0510", "title": "时间裂缝", "tags": ["科幻"], "author": "A9", "ctr_prior": 0.11, "freshness": 0.7, "hot": 540},
        {"id": "mv_9901", "title": "星际边界", "tags": ["科幻", "冒险"], "author": "A10", "ctr_prior": 0.16, "freshness": 0.8, "hot": 880},
        {"id": "mv_8802", "title": "银河饭店", "tags": ["科幻", "轻松"], "author": "A11", "ctr_prior": 0.14, "freshness": 0.6, "hot": 510},
        {"id": "mv_7703", "title": "山海小馆", "tags": ["美食", "治愈"], "author": "A12", "ctr_prior": 0.19, "freshness": 0.9, "hot": 690},
        {"id": "mv_6604", "title": "深夜食堂新篇", "tags": ["美食", "都市"], "author": "A13", "ctr_prior": 0.17, "freshness": 0.7, "hot": 730},
        {"id": "mv_5505", "title": "冠军之路", "tags": ["体育", "热血"], "author": "A14", "ctr_prior": 0.13, "freshness": 0.8, "hot": 640},
        {"id": "mv_4406", "title": "周末篮球场", "tags": ["体育", "轻松"], "author": "A15", "ctr_prior": 0.07, "freshness": 0.5, "hot": 180},
    ]


def default_scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="classic",
            description="stable profile + explicit query",
            tasks=[
                BenchTask("u_mystery", "想看悬疑推理", "feed_home", ["mv_8821", "mv_5511", "mv_1207"], {"悬疑": 0.9, "推理": 0.6}),
                BenchTask("u_food", "下饭治愈美食", "feed_home", ["mv_7703", "mv_6604", "mv_3010"], {"美食": 0.8, "治愈": 0.5}),
                BenchTask("u_sci", "轻松一点的科幻", "feed_home", ["mv_8802", "mv_9901", "mv_0510"], {"科幻": 0.7, "轻松": 0.4}),
            ],
        ),
        Scenario(
            name="cold_start",
            description="no profile; must blend query and hot fallback",
            tasks=[
                BenchTask("u_new_1", "热血体育", "feed_home", ["mv_5505", "mv_4406"], {}),
                BenchTask("u_new_2", "都市悬疑", "search", ["mv_8821", "mv_7102", "mv_1207"], {}),
                BenchTask("u_new_3", "轻松喜剧", "feed_home", ["mv_2008", "mv_6010", "mv_0904"], {}),
            ],
        ),
        Scenario(
            name="evolving_interest",
            description="profile says one thing, query reveals fresh intent",
            tasks=[
                BenchTask("u_shift_1", "最近想看科幻冒险", "feed_home", ["mv_9901", "mv_8802", "mv_0510"], {"悬疑": 0.8}),
                BenchTask("u_shift_2", "今天想看治愈美食", "feed_home", ["mv_7703", "mv_6604", "mv_3010"], {"体育": 0.7}),
                BenchTask("u_shift_3", "轻松篮球周末", "feed_home", ["mv_4406", "mv_5505", "mv_2008"], {"科幻": 0.8}),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def hit_rate_at_k(recommended: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(item_id in rel for item_id in recommended[:k]) else 0.0


def ndcg_at_k(recommended: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    dcg = 0.0
    for idx, item_id in enumerate(recommended[:k], start=1):
        if item_id in rel:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def coverage(recommended_lists: Iterable[Sequence[str]], corpus_size: int) -> float:
    seen = {item_id for recs in recommended_lists for item_id in recs if not item_id.startswith("ad_")}
    return len(seen) / corpus_size if corpus_size else 0.0


def intra_list_diversity(recommended: Sequence[str], corpus_index: Dict[str, Dict[str, Any]]) -> float:
    item_ids = [x for x in recommended if x in corpus_index]
    if len(item_ids) < 2:
        return 0.0
    pairs = 0
    distance_sum = 0.0
    for i in range(len(item_ids)):
        for j in range(i + 1, len(item_ids)):
            a = set(corpus_index[item_ids[i]].get("tags", []))
            b = set(corpus_index[item_ids[j]].get("tags", []))
            union = a | b
            sim = len(a & b) / len(union) if union else 0.0
            distance_sum += 1.0 - sim
            pairs += 1
    return distance_sum / pairs if pairs else 0.0


def trace_cost(trace: List[Dict[str, Any]]) -> float:
    # Lightweight proxy: each agent step costs 1, each veto costs 2 extra.
    return float(len(trace) + 2 * sum(1 for ev in trace if ev.get("action") == "veto"))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def hot_baseline(corpus: List[Dict[str, Any]], top_k: int) -> Callable[[BenchTask], List[Item]]:
    ranked = sorted(corpus, key=lambda x: -x.get("hot", 0))

    def run(_: BenchTask) -> List[Item]:
        return [Item(id=x["id"], score=float(x.get("hot", 0)), features=x, source="hot_baseline") for x in ranked[:top_k]]

    return run


def tag_baseline(corpus: List[Dict[str, Any]], top_k: int) -> Callable[[BenchTask], List[Item]]:
    def run(task: BenchTask) -> List[Item]:
        query = task.query.lower()
        out: List[Item] = []
        for item in corpus:
            tags = item.get("tags", [])
            score = 0.0
            score += sum(1.0 for tag in tags if tag.lower() in query)
            score += sum(task.profile_tags.get(tag, 0.0) for tag in tags)
            if score > 0:
                out.append(Item(id=item["id"], score=score, features=item, source="tag_baseline"))
        out.sort(key=lambda x: -x.score)
        return out[:top_k]

    return run


def agentic_runner(
    corpus: List[Dict[str, Any]],
    top_k: int,
    enable_collaboration: bool = True,
    adaptive_collaboration: bool = True,
) -> Callable[[BenchTask], tuple[List[Item], float, int, float]]:
    def run(task: BenchTask) -> tuple[List[Item], float, int, float]:
        pipe = AgenticPipeline(corpus=corpus, top_n=top_k,
                               enable_collaboration=enable_collaboration,
                               adaptive_collaboration=adaptive_collaboration)
        if task.profile_tags:
            pipe.memory.update_profile(task.user_id, tags=task.profile_tags)
        result = pipe.run(task.query, user_id=task.user_id, scene=task.scene)
        return result.items, result.total_ms, len(result.trace), trace_cost(result.trace)

    return run


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------
def evaluate_agentic(
    corpus: List[Dict[str, Any]],
    scenarios: List[Scenario],
    top_k: int = 5,
    enable_collaboration: bool = True,
    adaptive_collaboration: bool = True,
) -> List[RunRow]:
    rows: List[RunRow] = []
    run = agentic_runner(corpus, top_k, enable_collaboration=enable_collaboration,
                         adaptive_collaboration=adaptive_collaboration)
    for scenario in scenarios:
        for task in scenario.tasks:
            items, latency_ms, trace_steps, cost = run(task)
            rows.append(RunRow(
                scenario=scenario.name,
                user_id=task.user_id,
                query=task.query,
                recommended=[item.id for item in items],
                relevant=task.relevant_ids,
                latency_ms=latency_ms,
                trace_steps=trace_steps,
                trace_cost=cost,
            ))
    return rows


def evaluate_baseline(
    name: str,
    runner: Callable[[BenchTask], List[Item]],
    scenarios: List[Scenario],
    top_k: int,
) -> List[RunRow]:
    rows: List[RunRow] = []
    for scenario in scenarios:
        for task in scenario.tasks:
            items = runner(task)
            rows.append(RunRow(
                scenario=scenario.name,
                user_id=task.user_id,
                query=task.query,
                recommended=[item.id for item in items[:top_k]],
                relevant=task.relevant_ids,
                latency_ms=0.0,
                trace_steps=0,
                trace_cost=0.0,
            ))
    return rows


def summarize(rows: List[RunRow], corpus: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    corpus_index = {item["id"]: item for item in corpus}
    grouped: Dict[str, List[RunRow]] = defaultdict(list)
    for row in rows:
        grouped[row.scenario].append(row)

    def summary_for(part: List[RunRow]) -> Dict[str, float]:
        return {
            f"hit_rate@{top_k}": round(mean(hit_rate_at_k(r.recommended, r.relevant, top_k) for r in part), 4),
            f"ndcg@{top_k}": round(mean(ndcg_at_k(r.recommended, r.relevant, top_k) for r in part), 4),
            "coverage": round(coverage((r.recommended for r in part), len(corpus)), 4),
            "diversity": round(mean(intra_list_diversity(r.recommended, corpus_index) for r in part), 4),
            "latency_ms": round(mean(r.latency_ms for r in part), 4),
            "trace_steps": round(mean(r.trace_steps for r in part), 4),
            "trace_cost": round(mean(r.trace_cost for r in part), 4),
        }

    return {
        "overall": summary_for(rows),
        "by_scenario": {name: summary_for(part) for name, part in grouped.items()},
    }


def run_benchmark(top_k: int = 5) -> Dict[str, Any]:
    corpus = default_corpus()
    scenarios = default_scenarios()
    methods = {
        "AgenticRec-Gated": evaluate_agentic(corpus, scenarios, top_k, enable_collaboration=True, adaptive_collaboration=True),
        "AgenticRec-Collab": evaluate_agentic(corpus, scenarios, top_k, enable_collaboration=True, adaptive_collaboration=False),
        "AgenticRec-Core": evaluate_agentic(corpus, scenarios, top_k, enable_collaboration=False),
        "HotBaseline": evaluate_baseline("HotBaseline", hot_baseline(corpus, top_k), scenarios, top_k),
        "TagBaseline": evaluate_baseline("TagBaseline", tag_baseline(corpus, top_k), scenarios, top_k),
    }
    return {
        "top_k": top_k,
        "corpus_size": len(corpus),
        "tasks": sum(len(s.tasks) for s in scenarios),
        "methods": {
            name: {
                "summary": summarize(rows, corpus, top_k),
                "rows": [row.__dict__ for row in rows],
            }
            for name, rows in methods.items()
        },
    }


def print_report(report: Dict[str, Any]) -> None:
    k = report["top_k"]
    print(f"AgenticRec-Bench | tasks={report['tasks']} corpus={report['corpus_size']} top_k={k}\n")
    metric_names = [f"hit_rate@{k}", f"ndcg@{k}", "coverage", "diversity", "latency_ms", "trace_steps", "trace_cost"]
    header = "method".ljust(14) + " ".join(name.rjust(12) for name in metric_names)
    print(header)
    print("-" * len(header))
    for method, payload in report["methods"].items():
        overall = payload["summary"]["overall"]
        values = " ".join(str(overall[name]).rjust(12) for name in metric_names)
        print(method.ljust(14) + values)
    print("\nBy scenario")
    for method, payload in report["methods"].items():
        print(f"\n[{method}]")
        for scenario, metrics in payload["summary"]["by_scenario"].items():
            short = ", ".join(f"{m}={metrics[m]}" for m in metric_names[:4])
            print(f"  {scenario}: {short}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgenticRec-Bench.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print JSON instead of table")
    args = parser.parse_args()
    report = run_benchmark(top_k=args.top_k)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
