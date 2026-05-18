"""LLM backbone abstraction.

Real backbones (OpenAI / Qwen / DeepSeek) plug in via subclassing BaseLLM.
MockLLM is a deterministic, dependency-free stand-in for tests and demos.
"""
from __future__ import annotations

from typing import Any, Dict, List


class BaseLLM:
    name: str = "base"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Rule-based fake LLM. Lets the whole framework run without any API key.

    It inspects the *last* user message and returns a short, deterministic
    'thought' string that downstream agents can parse. Real backbones can
    replace it without changing any agent logic.
    """

    name = "mock"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        last = messages[-1]["content"] if messages else ""
        low = last.lower()
        if "recall" in low or "召回" in last:
            return "use vector+tag, weight=0.6/0.4 (query is mid-complexity)"
        if "rank" in low or "粗排" in last:
            if "skip_if_few" in low:
                return "skip (candidate set is small enough)"
            return "use lite-tower, top_k=200"
        if "rerank" in low or "重排" in last:
            return "boost fresh items, dedup by author, insert 1 ad slot"
        if "explain" in low or "解释" in last:
            return "matched user's recent tags + collaborative signal"
        if "critic" in low or "审查" in last:
            return "ok"
        return "ok"
