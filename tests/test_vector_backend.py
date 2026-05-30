"""Tests for Stage 5 vector backend adapters."""
from agentic_rec import AgenticPipeline, InMemoryVectorBackend
from agentic_rec.bench import default_corpus
from agentic_rec.tools import VectorTool


def test_vector_tool_uses_in_memory_backend_source():
    tool = VectorTool(default_corpus(), backend=InMemoryVectorBackend())
    items = tool(query="科幻 冒险", top_k=3)

    assert items
    assert items[0].source == "vector:in_memory_vector"
    assert all(item.features.get("id") == item.id for item in items)


def test_pipeline_accepts_custom_vector_backend():
    backend = InMemoryVectorBackend()
    pipe = AgenticPipeline(corpus=default_corpus(), top_n=5, vector_backend=backend)
    result = pipe.run("科幻冒险", user_id="u1")

    assert result.items
    assert any(event["agent"] == "RecallAgent" for event in result.trace)
    assert pipe.vector_backend is backend


if __name__ == "__main__":
    test_vector_tool_uses_in_memory_backend_source()
    test_pipeline_accepts_custom_vector_backend()
    print("vector backend ok")
