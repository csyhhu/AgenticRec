"""Tests for Stage 3 collaborative agents."""
from agentic_rec import AgenticPipeline
from agentic_rec.bench import default_corpus


def test_collaboration_agent_appears_in_trace():
    pipe = AgenticPipeline(corpus=default_corpus(), top_n=5, enable_collaboration=True)
    pipe.memory.update_profile("u1", tags={"科幻": 0.8})
    result = pipe.run("最近想看科幻冒险", user_id="u1")
    agents = [event["agent"] for event in result.trace]
    assert "CollaborationAgent" in agents
    assert any(item.features.get("collab_score") for item in result.items)


def test_collaboration_can_be_disabled():
    pipe = AgenticPipeline(corpus=default_corpus(), top_n=5, enable_collaboration=False)
    result = pipe.run("最近想看科幻冒险", user_id="u1")
    agents = [event["agent"] for event in result.trace]
    assert "CollaborationAgent" not in agents


if __name__ == "__main__":
    test_collaboration_agent_appears_in_trace()
    test_collaboration_can_be_disabled()
    print("collab ok")
