"""Minimal-token DeepSeek API smoke test.

Cost is trivial: 1 system token + 6 prompt tokens + 1 output token ≈ < 0.001 CNY.

Set DEEPSEEK_API_KEY in your environment before running:
    export DEEPSEEK_API_KEY="sk-xxx"
    python -m pytest tests/test_deepseek_api.py -v
"""
import os

import pytest

from agentic_rec import DeepSeekLLM


@pytest.fixture
def api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def test_chat_minimal(api_key):
    """Single short message, max_tokens=1 — cheapest possible call."""
    llm = DeepSeekLLM(api_key=api_key, max_tokens=1, temperature=0)
    reply = llm.chat([{"role": "user", "content": "1+1="}])
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_chat_short(api_key):
    """Barely more tokens — verifies reasonable response."""
    llm = DeepSeekLLM(api_key=api_key, max_tokens=4, temperature=0)
    reply = llm.chat([{"role": "user", "content": "say yes"}])
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_missing_key_raises():
    """Calling chat with no key set should raise ValueError."""
    llm = DeepSeekLLM(api_key="", model="deepseek-chat")
    with pytest.raises(ValueError, match="DeepSeek API key not set"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_model_default():
    llm = DeepSeekLLM(api_key="sk-test", model="deepseek-reasoner")
    assert llm.model == "deepseek-reasoner"
    assert llm.name == "deepseek"
