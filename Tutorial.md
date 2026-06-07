# AgenticRec Walkthrough

## 12. quickstart.py 执行流程

当运行 `python examples/quickstart.py` 时，系统经历了 6 个 Stage：

### Stage 1：Recall（召回）

RecallAgent 读取用户 `u_42` 的画像（`{"悬疑": 0.8, "轻松": 0.4}`），判断非冷启动。调用 vector + tag 两路工具进行召回。召回完成后检查候选集大小，若 < 30 则追加 hot 拓宽。

### Stage 2：Rank（粗排）

RankAgent 检查候选集大小。若 ≤ 200，直接跳过粗排（输出 `skip`），将候选集原样传递给下一阶段。

### Stage 3：IntentGate（协同闸门）

IntentGate 分析 query "想看点轻松的悬疑剧" 与用户画像的匹配情况：
- 画像中有"悬疑"和"轻松"标签
- query 中同样包含"悬疑"和"轻松"
- 判断为 `classic` 场景，跳过协同层

### Stage 4：Collaboration（协同）

当 IntentGate 判定需要协同时，CollaborationAgent 会：
1. 招募 top 3 相似用户
2. 选择 top 6 候选物品
3. SimilarUserAgent 和 ItemAgent 并行投票
4. 将协同分与原始分融合（α=0.35）

若 IntentGate 判定跳过，此阶段不执行。

### Stage 5：Rerank（重排）

RerankAgent 调用 BizRuleTool：
1. 同作者去重（最多保留 2 个）
2. feed_home 场景新鲜度提权（+15%）
3. 在第 4 位插入广告位
4. 按最终分数排序

### Stage 6：Critic（审查）+ Explain（解释）

CriticAgent 检查：
- 标签集中度是否 > 70%
- 广告比例是否 > 20%
- 若异常则否决，清空用户 tags 重新从 Stage 1 开始

ExplainAgent 为每个 item 生成可读解释：
- 广告 → "商业化插入位"
- 标签命中 → "命中你常看的悬疑,轻松（来源:vector:in_memory_vector）"
- 无命中 → "来源:hot, 评分 0.02"

---

## 13. 深度讨论：Agent 是包装还是真 Agent？

这是一个核心问题：**把 Pipeline 的每个模块用 Agent 包装一下，就能称为 Agentic 了吗？**

### 13.1 诚实评估

| 模块 | Agent 特性 | 详细分析 |
|------|-----------|---------|
| **RecallAgent** | ⚠️ 部分 | LLM 路由 + 反思（不足时拓宽），但本质是工具编排 |
| **RankAgent** | ❌ 包装 | 条件跳过 + 简单打分，无 LLM/反思/多策略 |
| **RerankAgent** | ❌ 包装 | BizRuleTool 的薄封装 |
| **ExplainAgent** | ⚠️ 部分 | 规则式生成，但为 LLM 生成留了接口 |
| **CollaborationAgent** | ✅ 真 Agent | 多智能体招募+投票+融合，有明确的多 Agent 协同 |
| **CriticAgent** | ✅ 真 Agent | 否决+触发反思重跑，完整的审查-修正循环 |
| **IntentGate** | ⚠️ 有新意 | 自适应决策，但当前是确定性规则 |

### 13.2 包装 vs 真 Agent 的分界线

一个模块是"真 Agent"而非"包装"的判断标准：

1. **是否有自主决策**：不只是执行固定规则，而是根据上下文选择策略
2. **是否有反思能力**：能检查自己的输出并修正
3. **是否有工具使用**：能动态选择和调用工具
4. **是否有 LLM 驱动的推理**：用 LLM 做 reasoning，而非纯规则
5. **是否有多 Agent 交互**：Agent 之间有协商、辩论、投票

按这个标准：
- CollaborationAgent + SimilarUserAgent + ItemAgent：满足 1、3、5
- CriticAgent：满足 1、2
- IntentGate：满足 1（有潜力升级到 4）
- RecallAgent/RankAgent/RerankAgent/ExplainAgent：主要满足 3

### 13.3 项目的真正价值

AgenticRec 的真正创新不在于把 Pipeline 的每个模块叫做 Agent，而在于：

1. **多智能体协同**：SimilarUserAgent 投票 + ItemAgent 自辩，这是传统搜推系统没有的范式
2. **否决反思机制**：CriticAgent 不是"打分"，而是"审查"——发现问题后触发整个管线重跑
3. **自适应闸门**：IntentGate 根据场景动态决定是否启用协同，避免不必要的开销
4. **可观测性**：每个 Agent 的 thought 都被 Trace 记录，让"为什么推荐这个"变得可追溯

---

## 14. 什么是 Agentic Recommendation？

### 14.1 分级定义

| Level | 名称 | 特征 | AgenticRec 对应 |
|-------|------|------|----------------|
| **Level 0** | 传统推荐 | 固定管线，纯模型/规则驱动 | HotBaseline, TagBaseline |
| **Level 1** | 弱 Agentic | 模块用 Agent 包装，有工具调用 | RecallAgent, RankAgent, RerankAgent |
| **Level 2** | 中 Agentic | 多智能体协同，有审查/反思 | CollaborationAgent, CriticAgent |
| **Level 3** | 强 Agentic | LLM 驱动的动态规划，Agent 辩论 | AgenticRec 的改进方向 |
| **Level 4** | 完全 Agentic | 端到端 LLM 推理，自主探索与学习 | 前沿研究方向 |

AgenticRec 定位在 **Level 1-2 之间**，有向 Level 3 演进的明确路径。

### 14.2 Agentic Recommendation 的核心特征

1. **自适应决策**：不是写死的 if-else，而是根据上下文选择策略
2. **多 Agent 协同**：不同 Agent 从不同视角贡献信号
3. **反思与修正**：能检查输出质量并自我修正
4. **可解释性**：每个决策都有可追溯的理由
5. **工具使用**：Agent 能动态选择和组合工具

### 14.3 与传统搜推的关系

AgenticRec 不是要替代传统搜推管线，而是**在元决策层引入 Agentic 能力**：
- 底层工具（向量检索、特征服务、业务规则）保持不变
- 编排层从"写死的 Pipeline"升级为"可反思的 Agent 议会"
- 新增协同层（CollaborationAgent）是传统管线没有的能力

---

## 15. MovieLens-1M 公开数据集实验

AgenticRec 已完整迁移到 MovieLens-1M 公开数据集。本章记录从数据适配到评测分析的完整流程。

### 15.1 数据准备

#### 下载 MovieLens-1M

从 [grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/) 下载 `ml-1m.zip`，解压到 `~/.agentic_rec/ml-1m/`，包含三个文件：

| 文件 | 内容 | 格式 |
|------|------|------|
| `movies.dat` | 3,883 部电影，含 title 和 18 类 genres | `movieId::title::genres` |
| `ratings.dat` | 6,040 用户 × 1M 条评分 (1-5) | `userId::movieId::rating::timestamp` |
| `users.dat` | 用户人口统计信息 | (未使用) |

#### 验证数据加载

```python
from agentic_rec.datasets import MovieLensAdapter
adapter = MovieLensAdapter()
adapter.print_validation_report()
```

输出：
```
============================================================
  MovieLensAdapter Validation  [status: ok]
============================================================
--- Corpus ---
  Items: 3883
  Fields valid: True
  Top tags: {'Drama': 1603, 'Comedy': 1200, 'Action': 503, 'Thriller': 492, 'Romance': 471}

--- Neighbor Profiles ---
  Count: 18
  neighbor_action: 16 tags, 10 history
  ...

--- Scenarios ---
  classic: 199 tasks | stable profile + explicit query
  cold_start: 190 tasks | few interactions; rely on query + hot fallback
  evolving_interest: 178 tasks | profile says one thing, query reveals fresh intent
  Total tasks: 567

--- User Train Map ---
  Users: 6040
  Items/user: min=16 max=1851 avg=132.1

Validation PASSED
```

### 15.2 数据适配架构

#### 目录结构

```
agentic_rec/datasets/
├── __init__.py
├── base.py              # DatasetAdapter 基类 (normalise / _build_user_profile_tags)
└── movielens.py         # MovieLens-1M 适配器
```

#### MovieLensAdapter 核心接口

```
MovieLensAdapter(raw_path=None, min_rating=4.0, min_user_interactions=10, seed=42)
├── load_corpus() → List[Dict]              # 3883 items
├── load_scenarios() → List[Scenario]       # classic / cold_start / evolving_interest
├── build_neighbor_profiles() → List[AgentProfile]  # 18 profiles, 每个 genre 一个
├── build_user_train_map(max_users=500) → Dict      # 用于 ItemKNN baseline
├── validate() → Dict                       # 自检报告
└── print_validation_report()               # 人类可读的验证报告
```

#### 字段映射

| MovieLens 原始字段 | AgenticRec 字段 | 转换方式 |
|-------------------|----------------|---------|
| movieId | `id` | `f"ml_{movieId}"` |
| title | `title` | 直接映射 |
| genres | `tags` | `"Action\|Comedy"` → `["Action", "Comedy"]` |
| — | `author` | 首个 genre 作为代理 |
| 电影平均评分 | `ctr_prior` | 归一化到 [0,1] |
| 评分时间跨度 | `freshness` | 归一化并反转 (跨度大=旧=低新鲜度) |
| 被评分次数 | `hot` | `log1p(count)` 后归一化 |

#### 场景拆分策略

| 场景 | 用户选取逻辑 | task 数 | 特点 |
|------|------------|---------|------|
| `classic` | 交互数 ≥10，前后半段 genre 重叠 >50% | 199 | 画像稳定，query 与画像一致 |
| `cold_start` | 交互数 10~30 | 190 | 数据稀疏，依赖 query + hot fallback |
| `evolving_interest` | 交互数 ≥10，前后半段 genre 重叠 <50% | 178 | 画像说一套，query 揭示新意图 |

#### BenchTask 构造

- 80% 交互用于构建 `profile_tags`（genre 加权偏好），20% 作为 ground-truth
- 评分 ≥ 4 的 item 标记为 `relevant_ids`
- `query` 由用户 top-2 genre 生成自然语言模板（如 `"I want Action and Sci-Fi movies"`）
- 所有 task 共用 `scene="feed_home"`

### 15.3 评测方法

#### 6 个被评测方法

| 方法 | 类型 | 说明 |
|------|------|------|
| **AgenticRec-Gated** | AgenticRec | 协同+自适应闸门 (enable_collaboration=True, adaptive_collaboration=True) |
| **AgenticRec-Collab** | AgenticRec | 强制启用协同 (enable_collaboration=True, adaptive_collaboration=False) |
| **AgenticRec-Core** | AgenticRec | 纯核心管线，关闭协同 (enable_collaboration=False) |
| **ItemKNN** | Baseline | 协同过滤，item-item 余弦相似度 + 用户历史加权打分 |
| **TagBaseline** | Baseline | 基于 query + profile_tags 的简单 tag 匹配 |
| **HotBaseline** | Baseline | 全局热门排序 |

#### 评测指标

| 指标 | 含义 | 范围 |
|------|------|------|
| HR@5 | Top-5 中至少命中一个 relevant item 的 task 比例 | [0,1] |
| Recall@5 | Top-5 中命中的 relevant item 占该 task 全部 relevant 的比例 | [0,1] |
| MRR@5 | 首个命中 item 的倒数排名均值 | [0,1] |
| NDCG@5 | 考虑命中位置的折损累计增益 | [0,1] |
| Coverage | 推荐结果覆盖的 item 种类占 corpus 的比例 | [0,1] |
| Diversity | 推荐列表内 item 间 tag 不相似度的均值 (1 - Jaccard) | [0,1] |
| Latency(ms) | 单 task 平均耗时 | ms |
| GateRate | IntentGate 启用协同的比例 (仅 AgenticRec) | [0,1] |
| VetoRate | CriticAgent 否决率 (仅 AgenticRec) | [0,1] |

### 15.4 运行评测

#### 快速测试（30 tasks/场景）

```bash
python examples/run_movielens_bench.py --max-tasks 30
```

#### 完整评测

```bash
python examples/run_movielens_bench.py
```

#### 自定义参数

```bash
# 指定数据目录
python examples/run_movielens_bench.py --data-dir D:/ml-1m

# 自定义 top-K 和 task 数量
python examples/run_movielens_bench.py --top-k 10 --max-tasks 50

# 导出 JSON 结果
python examples/run_movielens_bench.py --json --output report.json
```

### 15.5 实验结果

**配置**：150 tasks (50/场景)，top_k=5，MovieLens-1M (3,883 items, 6,040 users)

#### 总体指标

| Method | HR@5 | Recall@5 | MRR@5 | NDCG@5 | Coverage | Diversity |
|--------|------|----------|-------|--------|----------|-----------|
| **ItemKNN** | **12.67%** | **2.39%** | **6.67%** | **3.27%** | 1.11% | 63.37% |
| HotBaseline | 11.33% | 2.22% | 6.17% | 2.89% | 0.13% | 62.17% |
| AgenticRec-Core | 9.33% | 1.55% | 3.89% | 1.98% | 3.06% | 78.93% |
| TagBaseline | 5.33% | 0.89% | 2.06% | 1.03% | 3.37% | 25.91% |
| AgenticRec-Collab | 0.67% | 0.01% | 0.13% | 0.09% | 6.16% | 50.19% |
| AgenticRec-Gated | 0.67% | 0.01% | 0.13% | 0.09% | 6.00% | 51.88% |

#### 各场景 HR@5

| Method | classic | cold_start | evolving_interest |
|--------|---------|------------|-------------------|
| ItemKNN | 14.00% | 16.00% | 8.00% |
| HotBaseline | 14.00% | 14.00% | 6.00% |
| AgenticRec-Core | 12.00% | 14.00% | 2.00% |
| TagBaseline | 10.00% | 2.00% | 4.00% |
| AgenticRec-Collab | 2.00% | 0.00% | 0.00% |
| AgenticRec-Gated | 2.00% | 0.00% | 0.00% |

#### AgenticRec 内部统计

| Method | GateRate | VetoRate | AvgSteps | AvgLatency |
|--------|----------|----------|----------|------------|
| AgenticRec-Gated | 98.0% | 88.7% | 2.5 | 117ms |
| AgenticRec-Collab | 100% | 88.7% | 2.5 | 133ms |
| AgenticRec-Core | 0% | 0% | 4.0 | 46ms |

### 15.6 结果分析

#### 1. ItemKNN 综合最优，HotBaseline 是强力基准

ItemKNN (HR@5=12.67%) 略优于 HotBaseline (11.33%)，差距不大。这反映了一个事实：在 MovieLens-1M 上，简单推荐热门高评分电影就是一个很强的 baseline。ItemKNN 的优势主要体现在 `cold_start` 和 `evolving_interest` 场景中，协同信号帮助填补了 sparse 用户的偏好盲区。

#### 2. AgenticRec-Core 表现尚可，接近 HotBaseline

AgenticRec-Core (HR@5=9.33%) 关闭了协同和闸门，仅保留 Recall → Rank → Rerank 核心管线。虽然 HR 略低于 HotBaseline，但：
- **Diversity 最高** (78.93%)，远超 ItemKNN (63.37%) 和 HotBaseline (62.17%)，说明 AgenticRec 的混合召回 (vector + tag + hot) 天然产出更多样化的推荐
- **Coverage 最高** (3.06%)，是 HotBaseline (0.13%) 的 23 倍

#### 3. AgenticRec-Gated/Collab 几乎全失效 — 核心问题分析

Gated 和 Collab 两个变体 HR@5 仅 0.67%，VetoRate 高达 88.7%。**根本原因是 CriticAgent 的 tag concentration 检查在 MovieLens 上过于敏感**：

- 在默认的 15-item 中文 corpus 上，每个 item 只有 1-2 个 tags，query 提取的 tags 能自然分散到多个 item 上
- MovieLens 的 genres 分布极不均匀（Drama 占 41%，Comedy 占 31%），用户 query 如 "I want Drama and Thriller movies" 召回的结果中 Drama 占比很容易超过 70% 阈值
- 每次 veto 触发后，CriticAgent 会清空 `pipe.memory` 中的 profile tags，导致重新召回时失去用户画像信号，只靠纯 query 匹配，进一步恶化结果
- Gated 变体的 IntentGate 几乎总是判定为 `classic`（因为英文 query 的 tag 与 profile 一致），失去了自适应跳过协同的机会

#### 4. TagBaseline 效果差 — 英文匹配问题

TagBaseline (HR@5=5.33%) 远低于 HotBaseline。原因是：
- query 是英文自然语言（如 "I want Action and Sci-Fi movies"），而 TagBaseline 做的是简单的子串匹配 (`tag.lower() in query`)
- 18 个英文 genre 标签无法覆盖 query 中可能的语义变体，匹配精度有限

#### 5. 场景差异显著

- **cold_start** 场景中 HotBaseline (14%) 和 ItemKNN (16%) 表现最好，说明在没有丰富用户画像的情况下，热门信号和协同信号更可靠
- **evolving_interest** 是所有方法的弱项，最高 HR@5 仅 8% (ItemKNN)，反映了"用户兴趣漂移"这一推荐系统经典难题
- AgenticRec-Core 在 classic (12%) 和 cold_start (14%) 场景表现接近 baseline，但在 evolving_interest (2%) 明显掉队

### 15.7 改进方向

基于实验结果，AgenticRec 在 MovieLens-1M 上的改进方向：

1. **CriticAgent 阈值自适应**：将固定的 70% concentration 阈值改为基于 corpus tag 分布的自适应阈值。例如：若 corpus 中某 tag 天然占比高 (如 Drama 41%)，则相应提高该 tag 的容忍阈值

2. **Veto 后的修复策略**：当前 veto 后直接清空 profile tags 过于激进。更好的做法是保留原始 profile 但降低权重，或仅移除浓度最高的 tag 而非全部清空

3. **英文 query 的 tag 提取**：`IntentGate._query_tags()` 当前依赖中文分词逻辑。在英文数据集上，tag 提取应使用更鲁棒的方式（如直接枚举 corpus 中所有 tag 做子串匹配）

4. **引入更强 baseline**：考虑添加 LightFM、BPR、Neural CF 等经典推荐算法作为 baseline，更全面地评估 AgenticRec 的相对表现

5. **ItemKNN 优化**：当前 ItemKNN 在 500 用户上做 item-item 相似度计算耗时 77s。可考虑使用稀疏矩阵运算加速

---

## 16. Related Works

AgenticRec 处于 Agentic Recommendation 这一新兴方向的交叉点。以下梳理与之相关的重要工作及本项目与它们的差异。

### 16.1 多智能体协同推荐

#### MACF / MACRec — Multi-Agent Collaborative Filtering

AgenticRec Stage 3 的 CollaborationAgent 直接借鉴了 MACF 的动态招募与中心协调思想：

- **核心机制**：将相似用户和候选物品实例化为 agent，让它们对候选 item 投票，再由 orchestrator 将协同分融合回排序列表
- **AgenticRec 的继承**：`SimilarUserAgent` 基于 neighbor profiles 投票、`ItemAgent` 基于 item 特征自辩、`CollaborationOrchestrator` 做加权融合（α=0.35）
- **AgenticRec 的扩展**：增加了 IntentGate 场景闸门，避免协同层在 classic 场景无脑介入，只在冷启动/兴趣漂移时才启用

#### RecAgent (Wang et al., 2023)

用 LLM-based agent 模拟用户行为生成合成推荐数据的框架：

- **核心思想**：让 LLM agent 扮演用户，在模拟环境中浏览、评分、交互，生成 training data
- **与 AgenticRec 的关系**：方向互补 — RecAgent 侧重 simulation-based recommendation（数据生成），AgenticRec 侧重 online serving（在线推理）
- **潜在结合点**：RecAgent 生成的合成数据可用于 AgenticRec 的 cold_start 场景评测

### 16.2 LLM-Driven Recommendation

#### InteRecAgent / RecMind

用单一强 LLM 作为推荐系统的中央控制器：

- **核心机制**：用户通过自然语言表达需求 → LLM 理解意图 → 调用搜索/过滤/排序等工具 → 返回推荐列表
- **与 AgenticRec 的差异**：
  - InteRecAgent 是 **单 LLM + 工具调用** 范式，LLM 是必须的核心
  - AgenticRec 是 **多 Agent 协同 + LLM 可选** 范式，MockLLM 下零 API 也能跑完整链路
  - AgenticRec 强调工具优先于模型，搜广推 90% 复杂度在数据/特征/规则，Agent 只是编排层

#### LLMRec / P5 / GPT4Rec — Generative Recommendation

将推荐任务重新表述为 sequence generation：

- **核心机制**：把 user-item 交互历史、item 描述等序列化为 prompt，让 LLM 直接生成推荐 item ID 列表
- **与 AgenticRec 的差异**：
  - 生成式方法的 item 来自 LLM 的 token 空间，存在幻觉和冷门 item 消失风险
  - AgenticRec 的 item 始终来自真实 corpus（通过 ToolRegistry 检索），LLM 只做编排决策

### 16.3 Agentic Recommendation 综述

#### Agent4Rec (Zhang et al., 2024)

一份系统性的 agentic recommendation 综述，将 agent 在推荐系统中的应用按环节分类：

| 环节 | Agent 应用 | AgenticRec 对应 |
|------|-----------|----------------|
| 用户建模 | Agent 模拟用户偏好演化 | Memory + profile_tags |
| 召回 | Agent 动态选择召回策略 | RecallAgent (vector + tag + hot 多路) |
| 排序 | Agent 自适应调整排序权重 | RankAgent (lite scoring) + RerankAgent (biz rules) |
| 协同 | Agent 间投票/协商 | CollaborationAgent (MACF-style) |
| 解释 | Agent 生成可读推荐理由 | ExplainAgent |
| 审查 | Agent 检查输出质量 | CriticAgent (veto + retry) |
