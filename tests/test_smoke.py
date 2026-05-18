"""Smoke test: ensure the council runs end-to-end."""
from agentic_rec import AgenticPipeline


def test_run_smoke():
    corpus = [
        {"id": f"i_{k}", "title": f"item {k}", "tags": ["悬疑" if k % 2 else "治愈"],
         "author": f"a{k % 3}", "ctr_prior": 0.1, "freshness": 0.5, "hot": 100 + k}
        for k in range(20)
    ]
    pipe = AgenticPipeline(corpus=corpus, top_n=5)
    pipe.memory.update_profile("u1", tags={"悬疑": 1.0})
    res = pipe.run(query="悬疑", user_id="u1")
    assert len(res.items) <= 5
    assert res.total_ms >= 0
    assert any(ev["agent"] == "RecallAgent" for ev in res.trace)
    assert any(ev["agent"] == "RerankAgent" for ev in res.trace)


if __name__ == "__main__":
    test_run_smoke()
    print("smoke ok")
