"""Tests for Stage 4 adaptive collaboration gate."""
from agentic_rec import AgenticPipeline
from agentic_rec.bench import default_corpus


def _agents(result):
    return [event["agent"] for event in result.trace]


def _gate_event(result):
    return next(event for event in result.trace if event["agent"] == "IntentGate")


def test_intent_gate_skips_collaboration_for_classic_intent():
    pipe = AgenticPipeline(corpus=default_corpus(), top_n=5, adaptive_collaboration=True)
    pipe.memory.update_profile("u1", tags={"悬疑": 0.9, "推理": 0.6})
    result = pipe.run("想看悬疑推理", user_id="u1")

    assert "IntentGate" in _agents(result)
    assert "CollaborationAgent" not in _agents(result)
    gate = _gate_event(result)
    assert gate["action"] == "skip_collaboration"
    assert gate["payload"]["scenario"] == "classic"


def test_intent_gate_enables_collaboration_for_intent_shift():
    pipe = AgenticPipeline(corpus=default_corpus(), top_n=5, adaptive_collaboration=True)
    pipe.memory.update_profile("u1", tags={"悬疑": 0.8})
    result = pipe.run("最近想看科幻冒险", user_id="u1")

    assert "IntentGate" in _agents(result)
    assert "CollaborationAgent" in _agents(result)
    gate = _gate_event(result)
    assert gate["action"] == "enable_collaboration"
    assert gate["payload"]["scenario"] == "intent_shift"


def test_adaptive_collaboration_can_be_disabled():
    pipe = AgenticPipeline(corpus=default_corpus(), top_n=5, adaptive_collaboration=False)
    pipe.memory.update_profile("u1", tags={"悬疑": 0.9, "推理": 0.6})
    result = pipe.run("想看悬疑推理", user_id="u1")

    assert "IntentGate" not in _agents(result)
    assert "CollaborationAgent" in _agents(result)
