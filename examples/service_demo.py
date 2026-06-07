"""Run a tiny AgenticRec JSON service.

Usage:
    PYTHONPATH=. python examples/service_demo.py

Then open:
    http://127.0.0.1:8765/recommend?query=科幻冒险&user_id=u1
"""
from agentic_rec import AgenticPipeline, AgenticRecService, serve
from agentic_rec.bench import default_corpus


if __name__ == "__main__":
    pipeline = AgenticPipeline(corpus=default_corpus(), top_n=5)
    service = AgenticRecService(pipeline)
    server = serve(service, host="127.0.0.1", port=8765)
    print("AgenticRec service running on http://127.0.0.1:8765")
    print("Try /recommend?query=科幻冒险&user_id=u1 or /traces")
    server.serve_forever()
