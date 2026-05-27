"""The five council members + the orchestrator.

Each agent is intentionally short — its job is to *route* tools, reflect on
the result, and produce a Decision. Replace MockLLM with a real backbone to
get real reasoning.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .collab import CollaborationOrchestrator, apply_collaboration_scores
from .core import AgentMessage, BaseAgent, Decision, Item
from .gating import IntentGate


# ---------------------------------------------------------------------------
class RecallAgent(BaseAgent):
    name = "RecallAgent"

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        query = ctx["query"]
        user_id = ctx["user_id"]
        scene = ctx.get("scene", "feed_home")

        prof = self.memory.profile_of(user_id)
        user_tags = prof.get("tags", {})
        is_cold = not user_tags

        # ----- LLM-driven routing thought (mockable) -----
        thought = self.llm.chat([{"role": "user",
                                  "content": f"recall route for query='{query}' scene={scene}"}]) \
            if self.llm else "default"

        items: List[Item] = []
        plan: List[str] = []

        # Hot fallback first if cold start
        if is_cold and "hot" in self.tools.names():
            items += self.tools.get("hot")(top_k=20)
            plan.append("hot:20")

        if "vector" in self.tools.names():
            v = self.tools.get("vector")(query=query, top_k=40)
            items += v
            plan.append(f"vector:{len(v)}")

        if "tag" in self.tools.names():
            t = self.tools.get("tag")(query=query, user_tags=user_tags, top_k=40)
            items += t
            plan.append(f"tag:{len(t)}")

        # Reflection: if too narrow, widen with hot
        unique = {i.id: i for i in items}
        if len(unique) < 30 and "hot" in self.tools.names() and not is_cold:
            for it in self.tools.get("hot")(top_k=30):
                unique.setdefault(it.id, it)
            plan.append("hot+30(reflection)")

        merged = list(unique.values())
        return Decision(agent=self.name, thought=f"{thought} | plan={plan}",
                        action="recall", payload=merged)


# ---------------------------------------------------------------------------
class RankAgent(BaseAgent):
    name = "RankAgent"
    SKIP_THRESHOLD = 200

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])
        # Reflection: skip if candidate set already small
        if len(items) <= self.SKIP_THRESHOLD:
            thought = f"skip coarse ranking, candidates={len(items)} <= {self.SKIP_THRESHOLD}"
            return Decision(agent=self.name, thought=thought, action="skip", payload=items)

        # Otherwise apply a lite scoring (combine source signals + features)
        feat = self.tools.get("feature") if "feature" in self.tools.names() else None
        if feat:
            ids = [it.id for it in items]
            extra = feat(item_ids=ids)
            for it in items:
                e = extra.get(it.id, {})
                it.score = 0.5 * it.score + 0.5 * float(e.get("ctr_prior", 0))
        items.sort(key=lambda x: -x.score)
        kept = items[: self.SKIP_THRESHOLD]
        return Decision(agent=self.name, thought=f"lite-rank, kept={len(kept)}",
                        action="rank", payload=kept)


# ---------------------------------------------------------------------------
class CollaborationAgent(BaseAgent):
    """Recruit similar-user and item agents to refine ranking.

    This is a lightweight MACF-style stage. It does not call a remote LLM;
    instead it exposes the agent recruitment and voting mechanics in a fully
    deterministic way so benchmark results remain reproducible.
    """

    name = "CollaborationAgent"

    def __init__(self, llm=None, tools=None, memory=None, neighbor_profiles=None) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.orchestrator = CollaborationOrchestrator(neighbor_profiles)

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])
        user_id = ctx["user_id"]
        profile = self.memory.profile_of(user_id)
        user_tags = profile.get("tags", {})
        report = self.orchestrator.run(items, user_tags=user_tags, query=ctx["query"])
        reranked = apply_collaboration_scores(items, report)
        payload = {
            "items": reranked,
            "report": {
                "recruited_users": report.recruited_users,
                "recruited_items": report.recruited_items,
                "item_scores": report.item_scores,
            },
        }
        return Decision(agent=self.name, thought=report.thought,
                        action="collaborate", payload=payload)


# ---------------------------------------------------------------------------
class RerankAgent(BaseAgent):
    name = "RerankAgent"

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])
        scene = ctx.get("scene", "feed_home")
        rule = self.tools.get("biz_rule") if "biz_rule" in self.tools.names() else None
        if rule:
            items = rule(items=items, scene=scene)
        thought = f"rerank by scene={scene}, dedup+freshness+ad"
        return Decision(agent=self.name, thought=thought, action="rerank", payload=items)


# ---------------------------------------------------------------------------
class ExplainAgent(BaseAgent):
    name = "ExplainAgent"

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])
        prof = self.memory.profile_of(ctx["user_id"])
        user_tags = set(prof.get("tags", {}).keys())
        for it in items:
            if it.features.get("is_ad"):
                it.explain = "商业化插入位"
                continue
            tags = set(it.features.get("tags", []))
            hit = list(user_tags & tags)
            src = it.source
            if hit:
                it.explain = f"命中你常看的 {','.join(hit[:2])}（来源:{src}）"
            else:
                it.explain = f"来源:{src}, 评分{it.score:.2f}"
        return Decision(agent=self.name, thought="annotate explanations",
                        action="explain", payload=items)


# ---------------------------------------------------------------------------
class CriticAgent(BaseAgent):
    """Distribution / bias guard. Vetoes pathological outputs."""

    name = "CriticAgent"

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])
        if not items:
            return Decision(agent=self.name, thought="empty result, veto",
                            action="veto", payload=True)
        # Check tag concentration
        from collections import Counter
        cats = Counter()
        for it in items:
            for t in it.features.get("tags", []):
                cats[t] += 1
        if cats:
            top, top_cnt = cats.most_common(1)[0]
            if top_cnt > 0.7 * len(items):
                return Decision(
                    agent=self.name,
                    thought=f"tag '{top}' over-concentrated ({top_cnt}/{len(items)}), veto",
                    action="veto",
                    payload=True,
                )
        # Ad ratio sanity
        ads = sum(1 for it in items if it.features.get("is_ad"))
        if ads > max(1, len(items) // 5):
            return Decision(agent=self.name, thought="ad ratio too high",
                            action="veto", payload=True)
        return Decision(agent=self.name, thought="distribution ok", action="pass",
                        payload=False)


# ---------------------------------------------------------------------------
class OrchestratorAgent(BaseAgent):
    """Council chair: routes the message flow across agents."""

    name = "OrchestratorAgent"

    def __init__(self, llm=None, tools=None, memory=None,
                 recall=None, rank=None, collab=None, rerank=None,
                 explain=None, critic=None, intent_gate: IntentGate | None = None,
                 max_retry: int = 1) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.recall = recall
        self.rank = rank
        self.collab = collab
        self.rerank = rerank
        self.explain = explain
        self.critic = critic
        self.intent_gate = intent_gate
        self.max_retry = max_retry

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        trace = ctx["trace"]

        for attempt in range(self.max_retry + 1):
            # 1) recall
            d_recall = self.recall.run(
                AgentMessage(self.name, self.recall.name, "request"), ctx)
            trace.add(d_recall)
            items = d_recall.payload

            # 2) rank
            d_rank = self.rank.run(
                AgentMessage(self.name, self.rank.name, "request",
                             content={"items": items}), ctx)
            trace.add(d_rank)
            items = d_rank.payload

            # 3) optional multi-agent collaboration, guarded by Stage 4 intent gate
            if self.collab is not None:
                gate_decision = None
                if self.intent_gate is not None:
                    profile = self.memory.profile_of(ctx["user_id"])
                    gate_decision = self.intent_gate.decide(
                        query=ctx["query"],
                        user_tags=profile.get("tags", {}),
                        items=items,
                        scene=ctx.get("scene", "feed_home"),
                    )
                    ctx["intent_gate"] = gate_decision
                    trace.add(Decision(
                        agent=self.intent_gate.name,
                        thought="; ".join(gate_decision.reasons),
                        action="enable_collaboration" if gate_decision.enable_collaboration else "skip_collaboration",
                        payload={
                            "enable_collaboration": gate_decision.enable_collaboration,
                            "scenario": gate_decision.scenario,
                            "confidence": gate_decision.confidence,
                            "signals": gate_decision.signals,
                        },
                    ))
                should_collaborate = gate_decision.enable_collaboration if gate_decision else True
                if should_collaborate:
                    d_collab = self.collab.run(
                        AgentMessage(self.name, self.collab.name, "request",
                                     content={"items": items}), ctx)
                    trace.add(d_collab)
                    items = d_collab.payload["items"]
                    ctx["collaboration"] = d_collab.payload["report"]

            # 4) rerank
            d_re = self.rerank.run(
                AgentMessage(self.name, self.rerank.name, "request",
                             content={"items": items}), ctx)
            trace.add(d_re)
            items = d_re.payload

            # 5) critic
            d_crit = self.critic.run(
                AgentMessage(self.name, self.critic.name, "critique",
                             content={"items": items}), ctx)
            trace.add(d_crit)
            if d_crit.payload is True and attempt < self.max_retry:
                # vetoed → reflect: bias profile to broaden recall, retry
                prof = self.memory.profile_of(ctx["user_id"])
                prof["tags"] = {}  # widen
                continue
            break

        # 6) explain
        d_exp = self.explain.run(
            AgentMessage(self.name, self.explain.name, "request",
                         content={"items": items}), ctx)
        trace.add(d_exp)
        items = d_exp.payload

        return Decision(agent=self.name, thought="council finished",
                        action="orchestrate", payload=items)
