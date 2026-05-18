# AgenticRec

> **From Pipeline to Council** — 一个把搜索推荐从"管线"重构为"议会"的轻量级智能体框架。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Stage: Alpha](https://img.shields.io/badge/stage-alpha-orange.svg)]()
[![Lines: <1200](https://img.shields.io/badge/core_lines-<1200-brightgreen.svg)]()

## 这是什么

传统搜广推系统是一条**串行管线**：召回 → 粗排 → 精排 → 重排。  
每一段是一个固定的算子，跑完就走，**不会反思、不会争辩、不会换工具**。

`AgenticRec` 把这条管线**翻译成五个 Agent**：

```
                  ┌──────────────────────────┐
                  │   OrchestratorAgent      │
                  │  (议会主席：路由 + 仲裁)  │
                  └──────────┬───────────────┘
                             │
       ┌──────────┬──────────┼──────────┬──────────┐
       │          │          │          │          │
  ┌────▼────┐ ┌──▼─────┐ ┌──▼─────┐ ┌──▼─────┐ ┌──▼──────┐
  │ Recall  │ │ Rank   │ │ Rerank │ │Explain │ │ Critic  │
  │ Agent   │ │ Agent  │ │ Agent  │ │ Agent  │ │ Agent   │
  └─────────┘ └────────┘ └────────┘ └────────┘ └─────────┘
       │          │          │          │          │
       └──────────┴──────┬───┴──────────┴──────────┘
                         │
                  ┌──────▼──────┐
                  │ ToolRegistry│
                  │ Vector/Feat │
                  │ /Biz/ABTest │
                  └─────────────┘
```

每个 Agent：
- 持有自己的**工具**（向量检索、特征服务、业务规则、A/B 实验）
- 持有自己的**短期记忆**（会话）和**长期记忆**（用户画像）
- 通过统一的 `AgentMessage` 通信
- 可在 `<thought>` 里**反思**自己的输出，并选择换策略或回炉

## 为什么搜广推需要 Agent

| 维度 | 传统管线 | AgenticRec |
|---|---|---|
| 召回策略 | 写死多路并行 | 由 RecallAgent 看 query/用户态**动态选**多路+权重 |
| 冷启动 | if-else 分支硬切 | OrchestratorAgent 路由到冷启专用子图 |
| 长尾意图 | 难以兜底 | CriticAgent 检测分布异常 → 回炉重召回 |
| 可解释 | 离线人工归因 | ExplainAgent 在线生成可读理由 |
| A/B 迭代 | 改代码、发版 | 改 Agent prompt 或 tool 即可 |

> **关键洞察**：搜广推系统真正的复杂度从不是模型本身，而是**"什么场景下用哪个模型/规则/兜底"的元决策**。  
> Agent 化的本质，是把这个元决策从**写死的 if-else** 升级为**可学习、可反思、可热更**的策略。

## 30 秒上手

```bash
pip install -e .
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

## 五个 Agent 各自做什么

### 1. RecallAgent — 召回议员
- 工具：`VectorTool` / `TagTool` / `KGTool` / `HotTool`
- 决策：根据 query 复杂度和用户态**动态选路 + 配权**
- 反思：看候选集多样性，多样性不足时主动开第二路

### 2. RankAgent — 粗排议员
- 工具：双塔模型 / 轻量 GBDT
- 决策：候选 < 阈值时**自动跳过**，节省时延
- 反思：分布异常（全是同 tag）时回炉

### 3. RerankAgent — 重排议员
- 工具：业务规则、打散、提权、商业化插入
- 决策：根据 scene（feed/搜索结果页/广告位）切策略
- 反思：检查是否违反业务硬约束

### 4. ExplainAgent — 解释议员
- 工具：item 元数据、用户长记忆
- 输出：每个 item 一句**给运营/给用户**的可读理由

### 5. CriticAgent — 监督议员（议会守门人）
- 不参与生成，只**审阅**整体输出
- 检查：分布偏置、意图漂移、commercial/内容比例
- 异常时**否决并触发重跑**

## 设计哲学

1. **轻量优先**：核心 < 1200 行 Python，无重依赖。  
2. **拒绝魔法**：每个 Agent 是一个普通 class，prompt 全部可见可改。  
3. **工具优先于模型**：搜广推 90% 的复杂度在数据/特征/规则，Agent 只是**编排层**。  
4. **离线可灰度**：所有 Agent 决策都打 trace，方便对比 A/B。  
5. **LLM 可选**：MockLLM 模式下不调任何外部 API，纯规则也能跑通完整链路。

## 路线图

- [x] 五 Agent 核心 + Mock LLM
- [x] 工具注册表 + 向量/特征/业务规则
- [x] 决策 trace 与可视化 dump
- [ ] OpenAI / 通义 / DeepSeek backbone 适配
- [ ] Faiss / Milvus 真实向量后端
- [ ] LangGraph / OpenAI-Agents-SDK 对接 Adapter
- [ ] 在线服务化（FastAPI）+ 离线评估（HitRate/NDCG）
- [ ] 工业场景蓝本：电商首页 / 短视频 Feed / 站内搜索

## 与已有框架的关系

| 框架 | 定位 | 与 AgenticRec |
|---|---|---|
| LangGraph / AutoGen | 通用多智能体编排 | 上游可插拔的 backbone |
| Lagent / SmolAgents | 极简 Agent loop | 设计风格借鉴 |
| RecBole / EasyRec | 推荐算法库 | 互补（前者是模型动物园，本项目是编排骨架）|
| Dify / Coze | 通用 Agent 平台 | 不重叠（通用 vs 搜广推垂直）|

## License

MIT — 自由商用、欢迎 PR。

## 引用

如果本项目对你的研究/系统设计有启发，欢迎引用：

```bibtex
@misc{agenticrec2026,
  title  = {AgenticRec: From Pipeline to Council, A Lightweight Agentic Framework for Search and Recommendation},
  author = {GuoXun},
  year   = {2026},
  url    = {https://github.com/<your-org>/AgenticRec}
}
```
