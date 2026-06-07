"""Stage 6 service layer: request-level trace API and replay.

This module intentionally uses only the Python standard library. It gives
AgenticRec a small online-service facade without forcing FastAPI or any web
framework into the dependency-free core.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from .pipeline import AgenticPipeline, PipelineResult


@dataclass
class TraceRecord:
    """One request-level record for online debugging and replay."""

    request_id: str
    query: str
    user_id: str
    scene: str
    items: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    total_ms: float
    created_at: float = field(default_factory=time.time)


class TraceStore:
    """In-memory trace store with bounded retention."""

    def __init__(self, max_records: int = 128) -> None:
        self.max_records = max_records
        self._records: Dict[str, TraceRecord] = {}
        self._order: List[str] = []

    def add(self, record: TraceRecord) -> None:
        if record.request_id not in self._records:
            self._order.append(record.request_id)
        self._records[record.request_id] = record
        while len(self._order) > self.max_records:
            old_id = self._order.pop(0)
            self._records.pop(old_id, None)

    def get(self, request_id: str) -> Optional[TraceRecord]:
        return self._records.get(request_id)

    def list(self) -> List[TraceRecord]:
        return [self._records[request_id] for request_id in reversed(self._order)]


class AgenticRecService:
    """Small service facade around AgenticPipeline."""

    def __init__(self, pipeline: AgenticPipeline, store: Optional[TraceStore] = None) -> None:
        self.pipeline = pipeline
        self.store = store or TraceStore()

    def recommend(self, query: str, user_id: str = "anon", scene: str = "feed_home") -> Dict[str, Any]:
        request_id = uuid.uuid4().hex[:12]
        result = self.pipeline.run(query=query, user_id=user_id, scene=scene)
        record = TraceRecord(
            request_id=request_id,
            query=query,
            user_id=user_id,
            scene=scene,
            items=_items_to_dicts(result),
            trace=result.trace,
            total_ms=round(result.total_ms, 4),
        )
        self.store.add(record)
        return asdict(record)

    def trace(self, request_id: str) -> Dict[str, Any]:
        record = self.store.get(request_id)
        if record is None:
            raise KeyError(f"trace not found: {request_id}")
        return asdict(record)

    def traces(self) -> List[Dict[str, Any]]:
        return [asdict(record) for record in self.store.list()]

    def replay(self, request_id: str) -> Dict[str, Any]:
        record = self.store.get(request_id)
        if record is None:
            raise KeyError(f"trace not found: {request_id}")
        return replay_trace(record)


def replay_trace(record: TraceRecord) -> Dict[str, Any]:
    """Turn a trace record into a compact debugging timeline."""

    timeline = []
    for idx, event in enumerate(record.trace, start=1):
        timeline.append({
            "step": idx,
            "agent": event.get("agent"),
            "action": event.get("action"),
            "thought": event.get("thought", ""),
            "ms": event.get("ms", 0.0),
        })
    return {
        "request_id": record.request_id,
        "query": record.query,
        "user_id": record.user_id,
        "scene": record.scene,
        "timeline": timeline,
        "final_items": record.items,
        "total_ms": record.total_ms,
        "trace_steps": len(record.trace),
    }


def serve(service: AgenticRecService, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Start a tiny JSON HTTP server for demos and local debugging."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/health":
                    self._json({"ok": True})
                elif parsed.path == "/recommend":
                    query = qs.get("query", [""])[0]
                    user_id = qs.get("user_id", ["anon"])[0]
                    scene = qs.get("scene", ["feed_home"])[0]
                    self._json(service.recommend(query=query, user_id=user_id, scene=scene))
                elif parsed.path == "/traces":
                    self._json({"traces": service.traces()})
                elif parsed.path.startswith("/replay/"):
                    self._json(service.replay(parsed.path.rsplit("/", 1)[-1]))
                else:
                    self._json({"error": "not found"}, status=404)
            except KeyError as exc:
                self._json({"error": str(exc)}, status=404)
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._json({"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def main() -> None:
    """CLI entry point for the demo service."""

    from .bench import default_corpus

    pipeline = AgenticPipeline(corpus=default_corpus(), top_n=5)
    service = AgenticRecService(pipeline)
    server = serve(service)
    print("AgenticRec service running on http://127.0.0.1:8765")
    print("Try /recommend?query=科幻冒险&user_id=u1 or /traces")
    server.serve_forever()


def _items_to_dicts(result: PipelineResult) -> List[Dict[str, Any]]:
    return [
        {
            "id": item.id,
            "score": round(item.score, 4),
            "source": item.source,
            "explain": item.explain,
            "features": dict(item.features),
        }
        for item in result.items
    ]
