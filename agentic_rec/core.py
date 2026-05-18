"""Core primitives: Agent / Message / Tool / Memory / Item / Trace.

Designed to be tiny — every concept here fits on one screen.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Item & Decision — what the system actually produces
# ---------------------------------------------------------------------------
@dataclass
class Item:
    id: str
    score: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    explain: str = ""
    source: str = ""  # which recall path produced it

    def with_score(self, s: float) -> "Item":
        self.score = s
        return self


@dataclass
class Decision:
    """An agent's structured output for one step."""
    agent: str
    thought: str = ""
    action: str = ""
    payload: Any = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# AgentMessage — unified inter-agent protocol
# ---------------------------------------------------------------------------
@dataclass
class AgentMessage:
    sender: str
    receiver: str
    intent: str                 # "request" | "respond" | "critique"
    content: Dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Memory — short-term (session) + long-term (user profile)
# ---------------------------------------------------------------------------
class Memory:
    def __init__(self) -> None:
        self.short: List[AgentMessage] = []
        self.long: Dict[str, Dict[str, Any]] = {}  # user_id -> profile

    def remember(self, msg: AgentMessage) -> None:
        self.short.append(msg)
        if len(self.short) > 64:
            self.short = self.short[-64:]

    def profile_of(self, user_id: str) -> Dict[str, Any]:
        return self.long.setdefault(user_id, {"history": [], "tags": {}})

    def update_profile(self, user_id: str, **patch: Any) -> None:
        prof = self.profile_of(user_id)
        prof.update(patch)


# ---------------------------------------------------------------------------
# Tool & ToolRegistry — what agents can actually do
# ---------------------------------------------------------------------------
class Tool:
    """A callable wrapped with a name + spec, registered into the council."""

    name: str = "tool"
    description: str = ""

    def __call__(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"tool not found: {name}")
        return self._tools[name]

    def names(self) -> List[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Trace — observable timeline of decisions across agents
# ---------------------------------------------------------------------------
class Trace:
    def __init__(self) -> None:
        self.events: List[Decision] = []

    def add(self, decision: Decision) -> None:
        self.events.append(decision)

    def dump(self) -> List[Dict[str, Any]]:
        return [
            {
                "agent": d.agent,
                "thought": d.thought,
                "action": d.action,
                "payload": _safe(d.payload),
                "ms": round(d.elapsed_ms, 2),
            }
            for d in self.events
        ]

    def total_ms(self) -> float:
        return sum(d.elapsed_ms for d in self.events)


def _safe(p: Any) -> Any:
    try:
        if isinstance(p, list):
            return [getattr(x, "id", x) for x in p[:8]]
        return p
    except Exception:
        return str(p)


# ---------------------------------------------------------------------------
# BaseAgent — every council member inherits from this
# ---------------------------------------------------------------------------
class BaseAgent:
    name: str = "agent"

    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,         # forward ref
        tools: Optional[ToolRegistry] = None,
        memory: Optional[Memory] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or Memory()

    # subclasses override
    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        raise NotImplementedError

    # convenience: timed run
    def run(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        t0 = time.perf_counter()
        d = self.step(msg, ctx)
        d.elapsed_ms = (time.perf_counter() - t0) * 1000
        d.agent = self.name
        return d
