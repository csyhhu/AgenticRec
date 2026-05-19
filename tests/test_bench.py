"""Benchmark tests for AgenticRec-Bench."""
from agentic_rec.bench import run_benchmark


def test_benchmark_report_shape():
    report = run_benchmark(top_k=5)
    assert report["tasks"] == 9
    assert report["corpus_size"] == 16
    assert set(report["methods"]) == {"AgenticRec", "HotBaseline", "TagBaseline"}
    metrics = report["methods"]["AgenticRec"]["summary"]["overall"]
    assert metrics["hit_rate@5"] >= 0
    assert metrics["ndcg@5"] >= 0
    assert metrics["coverage"] > 0
    assert metrics["trace_steps"] > 0


if __name__ == "__main__":
    test_benchmark_report_shape()
    print("bench ok")
