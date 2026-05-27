"""Stage 4 intent gate for adaptive collaboration.

The gate decides whether the collaborative user-item council should be invoked
for a request. It is intentionally deterministic so decisions can be replayed in
benchmarks and production traces.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence

from .core import Item


@dataclass
class GateDecision:
    """Structured decision emitted by the intent gate."""

    enable_collaboration: bool
    scenario: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)


class IntentGate:
    """Latency-aware gate for deciding when collaboration is worth the cost."""

    name = "IntentGate"

    def __init__(
        self,
        min_confidence: float = 0.55,
        force_scenarios: Iterable[str] | None = None,
        block_scenarios: Iterable[str] | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.force_scenarios = set(force_scenarios or {"cold_start", "intent_shift", "ambiguous"})
        self.block_scenarios = set(block_scenarios or {"classic"})

    def decide(
        self,
        query: str,
        user_tags: Dict[str, float],
        items: Sequence[Item],
        scene: str = "feed_home",
    ) -> GateDecision:
        query_tags = _query_tags(query, items)
        profile_tags = set(user_tags)
        candidate_tags = _candidate_tags(items)
        top_ratio = _top_tag_ratio(items)

        reasons: list[str] = []
        score = 0.0

        if not user_tags:
            score += 0.45
            reasons.append("cold_start:no_user_profile")

        if query_tags and profile_tags and not (query_tags & profile_tags):
            score += 0.40
            reasons.append("intent_shift:query_profile_mismatch")

        if len(query_tags) >= 2:
            score += 0.15
            reasons.append("multi_intent_query")

        if len(candidate_tags) >= 4:
            score += 0.10
            reasons.append("diverse_candidates")

        if top_ratio >= 0.60:
            score -= 0.20
            reasons.append("classic:dominant_candidate_tag")

        if scene == "search" and query_tags:
            score -= 0.10
            reasons.append("search:explicit_intent")

        score = max(0.0, min(1.0, score))
        scenario = self._scenario(score, user_tags, query_tags, profile_tags, candidate_tags, top_ratio)
        enabled = scenario in self.force_scenarios or (scenario not in self.block_scenarios and score >= self.min_confidence)

        return GateDecision(
            enable_collaboration=enabled,
            scenario=scenario,
            confidence=round(score, 4),
            reasons=reasons or ["classic:stable_profile"],
            signals={
                "query_tags": sorted(query_tags),
                "profile_tags": sorted(profile_tags),
                "candidate_tags": sorted(candidate_tags),
                "top_tag_ratio": round(top_ratio, 4),
                "scene": scene,
            },
        )

    def _scenario(
        self,
        score: float,
        user_tags: Dict[str, float],
        query_tags: set[str],
        profile_tags: set[str],
        candidate_tags: set[str],
        top_ratio: float,
    ) -> str:
        if not user_tags:
            return "cold_start"
        if query_tags and profile_tags and not (query_tags & profile_tags):
            return "intent_shift"
        if query_tags and profile_tags and (query_tags & profile_tags):
            return "classic"
        if len(query_tags) >= 2 or len(candidate_tags) >= 5:
            return "ambiguous"
        if top_ratio >= 0.60 or score < self.min_confidence:
            return "classic"
        return "adaptive"


def _query_tags(query: str, items: Sequence[Item]) -> set[str]:
    query_lower = query.lower()
    known_tags = {tag for item in items for tag in item.features.get("tags", [])}
    return {tag for tag in known_tags if tag.lower() in query_lower}


def _candidate_tags(items: Sequence[Item]) -> set[str]:
    return {tag for item in items for tag in item.features.get("tags", []) if not item.features.get("is_ad")}


def _top_tag_ratio(items: Sequence[Item]) -> float:
    tags = Counter(tag for item in items if not item.features.get("is_ad") for tag in item.features.get("tags", []))
    total = sum(tags.values())
    if not total:
        return 0.0
    return tags.most_common(1)[0][1] / total
