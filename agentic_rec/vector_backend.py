"""Stage 5 vector backend adapters.

This module gives AgenticRec a real vector-backend seam without adding heavy
runtime dependencies. The default backend is an in-memory ANN-style index with
hash embeddings; production users can implement the same interface for Faiss,
Milvus, or an internal vector service.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Protocol, Sequence


@dataclass
class VectorHit:
    """One vector retrieval result."""

    id: str
    score: float
    payload: Dict[str, Any]


class VectorBackend(Protocol):
    """Minimal vector backend protocol used by recall tools."""

    name: str

    def add(self, documents: Sequence[Dict[str, Any]]) -> None:
        """Index documents with at least id/title/tags fields."""

    def search(self, query: str, top_k: int = 50) -> List[VectorHit]:
        """Return top-k nearest documents for the query."""


class HashEmbedding:
    """Dependency-free text embedding for demos and tests.

    It is not intended to compete with production embeddings. Its purpose is to
    make vector backend behavior deterministic and runnable immediately after
    clone, while preserving the same integration shape as Faiss/Milvus.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        values = [0.0] * self.dim
        tokens = _tokens(text)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            values[idx] += sign
        return _normalize(values)


class InMemoryVectorBackend:
    """Small ANN-style vector backend implemented with brute-force cosine."""

    name = "in_memory_vector"

    def __init__(self, embedding: HashEmbedding | None = None) -> None:
        self.embedding = embedding or HashEmbedding()
        self._rows: List[tuple[str, List[float], Dict[str, Any]]] = []

    def add(self, documents: Sequence[Dict[str, Any]]) -> None:
        self._rows = []
        for doc in documents:
            text = _document_text(doc)
            self._rows.append((doc["id"], self.embedding.encode(text), dict(doc)))

    def search(self, query: str, top_k: int = 50) -> List[VectorHit]:
        query_vec = self.embedding.encode(query)
        scored = [
            VectorHit(id=item_id, score=_cosine(query_vec, vector), payload=payload)
            for item_id, vector, payload in self._rows
        ]
        scored.sort(key=lambda hit: -hit.score)
        return scored[:top_k]


class ExternalVectorBackend:
    """Base adapter for real vector services.

    Subclasses only need to implement `search`. `add` is optional because many
    production vector stores are built offline and queried online.
    """

    name = "external_vector"

    def add(self, documents: Sequence[Dict[str, Any]]) -> None:
        self._documents = list(documents)

    def search(self, query: str, top_k: int = 50) -> List[VectorHit]:
        raise NotImplementedError("Implement search() for your Faiss/Milvus/internal vector service")


class FaissVectorBackend(ExternalVectorBackend):
    """Faiss adapter placeholder with the same protocol as production backends."""

    name = "faiss_vector"


class MilvusVectorBackend(ExternalVectorBackend):
    """Milvus adapter placeholder with the same protocol as production backends."""

    name = "milvus_vector"


def _document_text(doc: Dict[str, Any]) -> str:
    tags = " ".join(str(tag) for tag in doc.get("tags", []))
    return f"{doc.get('title', '')} {tags} {doc.get('author', '')}"


def _tokens(text: str) -> List[str]:
    text = text.lower().strip()
    if not text:
        return []
    coarse = text.split()
    chars = [text[i:i + 2] for i in range(max(1, len(text) - 1))]
    return coarse + chars


def _normalize(values: Iterable[float]) -> List[float]:
    vec = list(values)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))
