"""Tests for Stage 6 service and replay layer."""
from agentic_rec import AgenticPipeline, AgenticRecService, TraceStore
from agentic_rec.bench import default_corpus


def test_service_records_trace_and_replays_request():
    service = AgenticRecService(
        AgenticPipeline(corpus=default_corpus(), top_n=5),
        store=TraceStore(max_records=4),
    )

    response = service.recommend("科幻冒险", user_id="u1", scene="feed_home")

    assert response["request_id"]
    assert response["items"]
    assert any(event["agent"] == "RecallAgent" for event in response["trace"])

    replay = service.replay(response["request_id"])
    assert replay["query"] == "科幻冒险"
    assert replay["trace_steps"] == len(response["trace"])
    assert replay["timeline"][0]["agent"] == "RecallAgent"
    assert replay["final_items"][0]["id"] == response["items"][0]["id"]


def test_trace_store_retains_recent_records_only():
    store = TraceStore(max_records=2)
    service = AgenticRecService(AgenticPipeline(corpus=default_corpus(), top_n=3), store=store)

    first = service.recommend("悬疑", user_id="u1")
    second = service.recommend("美食", user_id="u2")
    third = service.recommend("体育", user_id="u3")

    traces = service.traces()
    ids = [record["request_id"] for record in traces]
    assert third["request_id"] in ids
    assert second["request_id"] in ids
    assert first["request_id"] not in ids


if __name__ == "__main__":
    test_service_records_trace_and_replays_request()
    test_trace_store_retains_recent_records_only()
    print("service ok")
