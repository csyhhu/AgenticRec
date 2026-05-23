# AgenticRec

> **From Pipeline to Council** — A lightweight agentic framework for search and recommendation systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Stage: Alpha](https://img.shields.io/badge/stage-alpha-orange.svg)]()
[![Lines: <1200](https://img.shields.io/badge/core_lines-<1200-brightgreen.svg)]()

## What Is AgenticRec?

Traditional search and recommendation systems are usually built as a sequential pipeline:

```text
Recall -> Ranking -> Reranking -> Explanation / Monitoring
```

Each stage is a fixed operator. It runs, passes its output downstream, and rarely reflects, negotiates, or changes tools dynamically.

`AgenticRec` reframes this pipeline as a **collaborative council of specialized agents**. Stage 3 adds `CollaborationAgent`, a MACF-style layer that dynamically recruits similar-user agents and candidate-item agents:

```text
                  ┌──────────────────────────┐
                  │   OrchestratorAgent      │
                  │   routing + arbitration  │
                  └──────────┬───────────────┘
                             │
       ┌──────────┬──────────┼──────────┬──────────┐
       │          │          │          │          │
  ┌────▼────┐ ┌──▼─────┐ ┌──▼──────┐ ┌──▼─────┐ ┌──▼──────┐
  │ Recall  │ │ Rank   │ │ Collab  │ │ Rerank │ │ Critic  │
  │ Agent   │ │ Agent  │ │ Agent   │ │ Agent  │ │ Agent   │
  └─────────┘ └────────┘ └─────────┘ └────────┘ └─────────┘
       │          │          │          │          │
       └──────────┴──────┬───┴──────────┴──────────┘
                         │
                  ┌──────▼──────┐
                  │ ToolRegistry│
                  │ Vector/Feat │
                  │ /Biz/ABTest │
                  └─────────────┘
```

Each agent can own tools, memory, and decision logic. The goal is not to replace recommender models with LLMs, but to use agents as a **meta-decision layer** that decides which retrieval tools, ranking models, fallback strategies, business rules, and evaluation traces should be activated under different scenarios.

## Why Agentic Search and Recommendation?

The real complexity of search and recommendation systems is often not the model itself, but the meta-decision problem:

> Under which scenario should we use which model, tool, rule, fallback, or retry strategy?

AgenticRec makes this meta-decision layer explicit, observable, and extensible.

| Dimension | Conventional Pipeline | AgenticRec |
|---|---|---|
| Recall strategy | Fixed multi-channel recall | `RecallAgent` dynamically chooses tools and weights |
| Cold start | Hard-coded fallback rules | `OrchestratorAgent` routes to cold-start behavior |
| Long-tail intent | Difficult to recover | `CriticAgent` can veto and trigger retry |
| Explainability | Mostly offline attribution | `ExplainAgent` generates online explanations |
| A/B iteration | Code change and deployment | Prompt/tool/agent policy can be iterated independently |

## Quickstart

```bash
pip install -e .
python examples/quickstart.py
```

```python
from agentic_rec import AgenticPipeline, MockLLM

pipeline = AgenticPipeline(llm=MockLLM())

results = pipeline.run(
    query="light mystery drama",
    user_id="u_42",
    scene="feed_home",
)

for item in results.items:
    print(item.id, item.score, "<-", item.explain)
```

`MockLLM` lets the full workflow run without any external API key.

## Agent Roles

### 1. RecallAgent

- Tools: `VectorTool`, `TagTool`, `KGTool`, `HotTool`
- Dynamically selects retrieval paths based on query and user profile
- Reflects on candidate diversity and can activate additional recall tools

### 2. RankAgent

- Uses lightweight ranking tools or feature services
- Skips ranking when the candidate set is small enough
- Keeps latency-aware ranking behavior explicit

### 3. CollaborationAgent

- Dynamically recruits similar-user agents and candidate-item agents
- Lets recruited agents vote on candidate relevance
- Blends collaborative scores back into the ranked list

### 4. RerankAgent

- Applies scene-aware business rules
- Handles deduplication, freshness boost, diversification, and ad insertion
- Checks business constraints before final output

### 5. ExplainAgent

- Uses item metadata and user memory
- Produces readable reasons for each recommended item

### 6. CriticAgent

- Acts as a guardrail rather than a generator
- Checks distribution bias, intent drift, ad ratio, and other constraints
- Can veto a result and trigger a retry

## Design Principles

1. **Lightweight first**: the core framework stays small and dependency-free.
2. **No hidden magic**: agents are normal Python classes with explicit logic.
3. **Tools over models**: recommender systems depend heavily on data, features, and rules; agents orchestrate them.
4. **Observable by default**: all decisions are recorded as traces for replay, debugging, and A/B analysis.
5. **LLM-optional**: the whole pipeline can run with `MockLLM` for deterministic testing.

## AgenticRec-Bench

`AgenticRec` includes a zero-dependency evaluation loop: **AgenticRec-Bench**.

It is not a replacement for industrial offline evaluation, but a reproducible credibility layer for the framework.

- **3 scenarios**: `classic`, `cold_start`, `evolving_interest`
- **4 methods**: `AgenticRec-Collab`, `AgenticRec-Core`, `HotBaseline`, `TagBaseline`
- **7 metrics**: `HitRate@K`, `NDCG@K`, `Coverage`, `Diversity`, `Latency`, `TraceSteps`, `TraceCost`
- **9 tasks + 16 items**: no dataset download and no API key required

```bash
PYTHONPATH=. python -m agentic_rec.bench
# or after install
agentic-rec-bench --top-k 5
```

Example output:

```text
AgenticRec-Bench | tasks=9 corpus=16 top_k=5
method          hit_rate@5       ndcg@5     coverage    diversity   latency_ms  trace_steps   trace_cost
---------------------------------------------------------------------------------------------------------
AgenticRec-Collab      0.8889       0.7986          1.0       0.7099       0.2702            6          6.0
AgenticRec-Core      0.8889       0.7778          1.0        0.716       0.1288            5          5.0
HotBaseline         0.6667       0.2553       0.3125       0.8667          0.0            0          0.0
TagBaseline            1.0        0.907          1.0        0.663          0.0            0          0.0
```

This benchmark makes the core claim testable: AgenticRec not only produces recommendation lists, but also exposes decision traces that can be replayed and evaluated.

## Roadmap

- [x] Five-agent core with `MockLLM`
- [x] Tool registry with vector, feature, hot fallback, and business-rule tools
- [x] Decision trace dumping
- [x] Offline evaluation loop with HitRate/NDCG/Coverage/Diversity/Latency/TraceCost
- [x] Stage 3 collaborative agents: `SimilarUserAgent`, `ItemAgent`, and `CollaborationAgent`
- [ ] OpenAI / Qwen / DeepSeek backbone adapters
- [ ] Faiss / Milvus vector backend tools
- [ ] LangGraph / OpenAI Agents SDK adapters
- [ ] FastAPI service and trace dashboard
- [ ] Industrial blueprints for e-commerce, short-video feed, and site search

## Relation to Existing Frameworks

| Framework | Positioning | Relation to AgenticRec |
|---|---|---|
| LangGraph / AutoGen | General multi-agent orchestration | Can be used as upstream backbones |
| Lagent / SmolAgents | Minimal agent loops | Inspires the lightweight design |
| MACF / MACRec | Multi-agent recommendation | Stage 3 borrows dynamic recruitment and central coordination ideas |
| RecBole / EasyRec | Recommendation model libraries | Complementary: model zoo vs. orchestration layer |
| Dify / Coze | General agent platforms | Different scope: general-purpose vs. search/recommendation-specific |

## License

MIT.

## Citation

If this project helps your research or system design, please cite it as:

```bibtex
@misc{agenticrec2026,
  title        = {AgenticRec: From Pipeline to Council, A Lightweight Agentic Framework for Search and Recommendation},
  author       = {GuoXun},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/guoxun/AgenticRec},
  note         = {An agentic search and recommendation framework with tool orchestration, collaborative user/item agents, decision traces, and built-in benchmark evaluation}
}
```

Reference description:

> AgenticRec is a lightweight agentic framework for search and recommendation systems. It transforms the traditional recall-ranking-reranking pipeline into a council of specialized agents coordinated by an orchestrator. The framework emphasizes tool orchestration, collaborative user/item agents, optional LLM reasoning, observable decision traces, and built-in benchmark evaluation for classic, cold-start, and evolving-interest recommendation scenarios.
