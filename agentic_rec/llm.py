"""LLM backbone abstraction.

Real backbones (OpenAI / Qwen / DeepSeek) plug in via subclassing BaseLLM.
MockLLM is a deterministic, dependency-free stand-in for tests and demos.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


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


class DeepSeekLLM(BaseLLM):
    """DeepSeek API backbone. OpenAI-compatible, drop-in replacement for MockLLM.

    Args:
        api_key: DeepSeek API token (``sk-...``). If empty, reads ``DEEPSEEK_API_KEY`` env var.
        model: Model name (default ``deepseek-chat``, or ``deepseek-reasoner``).
        base_url: Override the API base URL.
        temperature: Sampling temperature (0–2).
        max_tokens: Max tokens in the response.
    """

    name = "deepseek"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        print(f"=== DeepSeek LLM initialized with model={model} ===")
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not set. Pass api_key= or set DEEPSEEK_API_KEY env var."
            )

        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }).encode("utf-8")

        req = Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=kwargs.get("timeout", 30)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            raise RuntimeError(f"DeepSeek API request failed: {e}") from e

        return data["choices"][0]["message"]["content"]
