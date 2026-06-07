# AgenticRec

> **From Pipeline to Council** — 一个把搜索推荐从"管线"重构为"议会"的轻量级智能体框架。

[English README](README_EN.md) | 中文说明

知乎发布文章：
把搜广推从「管线」重构成「议会」｜AgenticRec：一个 1200 行的轻量级智能体推荐框架
https://zhuanlan.zhihu.com/p/2039843956624725473


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Stage: Alpha](https://img.shields.io/badge/stage-alpha-orange.svg)]()
[![Lines: <1200](https://img.shields.io/badge/core_lines-<1200-brightgreen.svg)]()

## 这是什么

传统搜广推系统是一条**串行管线**：召回 → 粗排 → 精排 → 重排。  
每一段是一个固定的算子，跑完就走，**不会反思、不会争辩、不会换工具**。

`AgenticRec` 把这条管线**翻译成五个核心 Agent**，第五阶段加入可插拔 `VectorBackend`，第六阶段新增请求级 `Trace API` 与 `Replay`，让框架从离线 demo 走向可调试服务原型：

```
                  ┌──────────────────────────┐
                  │   OrchestratorAgent      │
                  │  (议会主席：路由 + 仲裁)  │
                  └──────────┬───────────────┘
                             │
       ┌──────────┬──────────┼──────────┬──────────┐
       │          │          │          │          │
  ┌────▼────┐ ┌──▼─────┐ ┌──▼──────┐ ┌──▼─────┐ ┌──▼──────┐
  │ Recall  │ │ Rank   │ │Intent │ │ Rerank │ │ Critic  │
  │ Agent   │ │ Agent  │ │ Gate   │ │ Agent  │ │ Agent   │
  └─────────┘ └────────┘ └─────────┘ └────────┘ └─────────┘
       │          │          │          │          │
       └──────────┴──────┬───┴──────────┴──────────┘
                         │
                  ┌──────▼──────┐
                  │ ToolRegistry│
                  │ VectorBackend│
                  │ Feat/BizRule │
                  └─────────────┘
```

`VectorBackend` 会把 `VectorTool` 和真实向量服务解耦：默认 `InMemoryVectorBackend` 零依赖可复现，生产侧可以替换为 Faiss、Milvus 或内部向量检索服务；`Trace API` 会把每次请求的候选、决策链和最终结果保存为可回放记录；`IntentGate` 继续负责决定是否启用协同议会。

每个 Agent：
- 持有自己的**工具**（向量检索、特征服务、业务规则、A/B 实验）
- 持有自己的**短期记忆**（会话）和**长期记忆**（用户画像）
- 通过统一的 `AgentMessage` 通信
- 可在 `<thought>` 里**反思**自己的输出，并选择换策略或回炉

## 为什么搜广推需要 Agent

| 维度 | 传统管线 | AgenticRec |
|---|---|---|
| 召回策略 | 写死多路并行 | RecallAgent 调用可插拔 `VectorBackend` + tag/hot 多路召回 |
| 冷启动 | if-else 分支硬切 | IntentGate 识别冷启 → 启用协同议会 |
| 兴趣漂移 | 难以兜底 | IntentGate 检测 query/profile mismatch → 启用相似用户与物料投票 |
| 可解释 | 离线人工归因 | ExplainAgent 在线生成可读理由 |
| A/B 迭代 | 改代码、发版 | 改 Agent prompt 或 tool 即可 |

> **关键洞察**：搜广推系统真正的复杂度从不是模型本身，而是**"什么场景下用哪个模型/规则/兜底"的元决策**。  
> Agent 化的本质，是把这个元决策从**写死的 if-else** 升级为**可学习、可反思、可热更**的策略。

## 30 秒上手

```bash
pip install -e .
# export DEEPSEEK_API_KEY=sk-YOUR_DS_KEY  # export your token here to enable the deepseek llm backbone
python examples/quickstart.py
```

```python
from agentic_rec import AgenticPipeline, MockLLM

pipeline = AgenticPipeline(llm=MockLLM())

results = pipeline.run(
    query="想看点轻松的悬疑剧",
    user_id="u_42",
    scene="feed_home",
)

for item in results.items:
    print(item.id, item.score, "←", item.explain)
```

输出（节选）：

```
mv_8821  0.913  ← 命中"悬疑+轻松"双意图，召回路径 vector+tag，粗排第3，重排提权(新剧加权)
mv_7102  0.881  ← 用户上周看完同导演作品，长记忆触发
...
[OrchestratorAgent] 本次决策：跳过粗排（候选<200直入精排），节省 12ms
```

## 六个议会角色各自做什么

### 1. RecallAgent — 召回议员
- 工具：`VectorTool` / `TagTool` / `KGTool` / `HotTool`
- 后端：`VectorTool` 可接 `InMemoryVectorBackend` / `FaissVectorBackend` / `MilvusVectorBackend`
- 决策：根据 query 复杂度和用户态**动态选路 + 配权**
- 反思：看候选集多样性，多样性不足时主动开第二路

### 2. RankAgent — 粗排议员
- 工具：双塔模型 / 轻量 GBDT
- 决策：候选 < 阈值时**自动跳过**，节省时延
- 反思：分布异常（全是同 tag）时回炉

### 3. IntentGate — 协同闸门
- 输入：query tags、用户画像、候选集 tag 分布、scene
- 决策：稳定意图跳过协同，冷启动/兴趣漂移/多意图启用 `CollaborationAgent`
- 价值：避免协同层在 classic 场景无脑介入，同时保留冷启动与兴趣漂移收益

### 4. CollaborationAgent — 协同议员
- 动态招募 `SimilarUserAgent` 与 `ItemAgent`
- 让相似用户和候选物料分别投票
- 将协同分 blend 回排序分，并写入 trace

### 5. RerankAgent — 重排议员
- 工具：业务规则、打散、提权、商业化插入
- 决策：根据 scene（feed/搜索结果页/广告位）切策略
- 反思：检查是否违反业务硬约束

### 6. ExplainAgent — 解释议员
- 工具：item 元数据、用户长记忆
- 输出：每个 item 一句**给运营/给用户**的可读理由

### 7. CriticAgent — 监督议员（议会守门人）
- 不参与生成，只**审阅**整体输出
- 检查：分布偏置、意图漂移、commercial/内容比例
- 异常时**否决并触发重跑**

## 设计哲学

1. **轻量优先**：核心 < 1200 行 Python，无重依赖。  
2. **拒绝魔法**：每个 Agent 是一个普通 class，prompt 全部可见可改。  
3. **工具优先于模型**：搜广推 90% 的复杂度在数据/特征/规则，Agent 只是**编排层**。  
4. **离线可灰度**：所有 Agent 决策都打 trace，方便对比 A/B。  
5. **LLM 可选**：MockLLM 模式下不调任何外部 API，纯规则也能跑通完整链路。

## 可插拔向量后端

第五阶段新增 `agentic_rec/vector_backend.py`，让召回层从 demo 逻辑走向可替换的工程接口：

```python
from agentic_rec import AgenticPipeline, InMemoryVectorBackend

pipeline = AgenticPipeline(
    vector_backend=InMemoryVectorBackend(),
)
```

内置后端：

- `InMemoryVectorBackend`：零依赖 hash embedding + cosine 检索，适合 demo、测试和 benchmark
- `FaissVectorBackend`：预留 Faiss adapter 接口，适合本地 ANN 检索
- `MilvusVectorBackend`：预留 Milvus adapter 接口，适合在线向量服务
- `ExternalVectorBackend`：给公司内部向量检索服务继承实现

## Trace API 与请求回放

第六阶段新增 `agentic_rec/service.py`，把离线 pipeline 包成一个零依赖 JSON 服务层：

```python
from agentic_rec import AgenticPipeline, AgenticRecService

service = AgenticRecService(AgenticPipeline())
response = service.recommend("科幻冒险", user_id="u1")
replay = service.replay(response["request_id"])
```

也可以启动本地调试服务：

```bash
agentic-rec-serve
# http://127.0.0.1:8765/recommend?query=科幻冒险&user_id=u1
# http://127.0.0.1:8765/traces
# http://127.0.0.1:8765/replay/{request_id}
```

`TraceStore` 会记录请求级 `query / user / scene / items / trace / total_ms`，`replay_trace()` 会把 Agent 决策链压成可读 timeline，适合做线上问题定位、A/B 样本复盘和 Trace Dashboard 原型。

## AgenticRec-Bench

`AgenticRec` 现在内置一个零依赖评测闭环：`AgenticRec-Bench`。

它不是工业离线评测的替代品，而是一个可执行的可信度层：

- **3 类场景**：`classic` / `cold_start` / `evolving_interest`
- **5 个方法**：`AgenticRec-Gated` / `AgenticRec-Collab` / `AgenticRec-Core` / `HotBaseline` / `TagBaseline`
- **7 个指标**：`HitRate@K`、`NDCG@K`、`Coverage`、`Diversity`、`Latency`、`TraceSteps`、`TraceCost`
- **9 个任务 + 16 个 item**：无需下载数据、无需 API key，克隆后即可复现

```bash
PYTHONPATH=. python -m agentic_rec.bench
# or after install
agentic-rec-bench --top-k 5
```

示例输出：

```text
AgenticRec-Bench | tasks=9 corpus=16 top_k=5
method          hit_rate@5       ndcg@5     coverage    diversity   latency_ms  trace_steps   trace_cost
---------------------------------------------------------------------------------------------------------
AgenticRec-Gated      0.8889       0.7986          1.0       0.6975       0.8359       6.6667       6.6667
AgenticRec-Collab      0.8889       0.7917          1.0       0.6975       1.4877            6          6.0
AgenticRec-Core      0.8889       0.7778          1.0       0.6852       1.3959            5          5.0
HotBaseline         0.6667       0.2553       0.3125       0.8667          0.0            0          0.0
TagBaseline            1.0        0.907          1.0        0.663          0.0            0          0.0
```

> 这让 AgenticRec 的核心主张可被验证：Agent 层不仅产出推荐列表，还产出可观测的决策 trace，可用于 A/B、回放和策略调试。

## 路线图

- [x] 五 Agent 核心 + Mock LLM
- [x] 工具注册表 + 向量/特征/业务规则
- [x] 决策 trace 与可视化 dump
- [x] 离线评测闭环（HitRate/NDCG/Coverage/Diversity/Latency/TraceCost）
- [x] OpenAI / 通义 / DeepSeek backbone 适配
- [x] 第三阶段多智能体协同（SimilarUserAgent / ItemAgent / CollaborationAgent）
- [x] 第四阶段自适应协同闸门（IntentGate / Gated Collaboration）
- [x] 第五阶段可插拔向量后端（VectorBackend / InMemory / Faiss / Milvus adapter seam）
- [x] 第六阶段请求级 Trace API + Replay（AgenticRecService / TraceStore / replay_trace）
- [ ] Faiss / Milvus 生产级索引接入示例
- [ ] LangGraph / OpenAI-Agents-SDK 对接 Adapter
- [ ] Trace Dashboard 可视化面板
- [ ] 工业场景蓝本：电商首页 / 短视频 Feed / 站内搜索

## 与已有框架的关系

| 框架 | 定位 | 与 AgenticRec |
|---|---|---|
| LangGraph / AutoGen | 通用多智能体编排 | 上游可插拔的 backbone |
| Lagent / SmolAgents | 极简 Agent loop | 设计风格借鉴 |
| MACF / MACRec | 多智能体协同推荐 | Stage 3 借鉴动态招募与中心协调思想，Stage 4 增加场景闸门避免全量调用 |
| Faiss / Milvus | 向量检索后端 | Stage 5 提供 `VectorBackend` adapter seam，可替换默认 in-memory 后端 |
| RecBole / EasyRec | 推荐算法库 | 互补（前者是模型动物园，本项目是编排骨架）|
| Dify / Coze | 通用 Agent 平台 | 不重叠（通用 vs 搜广推垂直）|

## License

MIT — 自由商用、欢迎 PR。

## 引用

如果本项目对你的研究/系统设计有启发，欢迎引用：

```bibtex
@misc{agenticrec2026,
  title        = {AgenticRec: From Pipeline to Council, A Lightweight Agentic Framework for Search and Recommendation},
  author       = {GuoXun},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/guoxun/AgenticRec},
  note         = {An agentic search and recommendation framework with pluggable vector backends, tool orchestration, adaptive collaboration gates, collaborative user/item agents, decision traces, and built-in benchmark evaluation}
}
```

English reference description:

> AgenticRec is a lightweight agentic framework for search and recommendation systems. It transforms the traditional recall-ranking-reranking pipeline into a council of specialized agents coordinated by an orchestrator. The framework emphasizes pluggable vector backends, tool orchestration, adaptive collaboration gates, collaborative user/item agents, optional LLM reasoning, observable decision traces, and built-in benchmark evaluation for classic, cold-start, and evolving-interest recommendation scenarios.
