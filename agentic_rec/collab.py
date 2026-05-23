"""Collaborative filtering agents for AgenticRec Stage 3.

This module adds a lightweight MACF-style workflow: instantiate similar users
and candidate items as agents, let them cast preference votes, and let a small
orchestrator merge those votes back into the ranking list.

The implementation is deterministic and dependency-free by design. It is meant
as a readable blueprint for dynamic agent recruitment, not as a replacement for
industrial collaborative filtering infrastructure.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from .core import Item


@dataclass
class PreferenceVote:
    """One agent's vote for one item."""

    agent: str
    item_id: str
    score: float
    reason: str


@dataclass
class AgentProfile:
    """Profile used to instantiate a lightweight user/item agent."""

    id: str
    tags: Dict[str, float] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationReport:
    """Structured output of a collaborative recommendation round."""

    recruited_users: List[str]
    recruited_items: List[str]
    votes: List[PreferenceVote]
    item_scores: Dict[str, float]
    thought: str


class SimilarUserAgent:
    """A recruited neighbor user who votes for candidate items."""

    def __init__(self, profile: AgentProfile) -> None:
        self.profile = profile
        self.name = f"SimilarUserAgent:{profile.id}"

    def vote(self, items: Sequence[Item], query_tags: Iterable[str]) -> List[PreferenceVote]:
        votes: List[PreferenceVote] = []
        q_tags = set(query_tags)
        for item in items:
            if item.features.get("is_ad"):
                continue
            tags = set(item.features.get("tags", []))
            profile_hit = sum(self.profile.tags.get(tag, 0.0) for tag in tags)
            query_hit = len(tags & q_tags) / max(1, len(q_tags))
            history_hit = 0.25 if item.id in self.profile.history else 0.0
            score = 0.55 * profile_hit + 0.30 * query_hit + history_hit
            if score > 0:
                reason = f"neighbor={self.profile.id}, shared_tags={','.join(sorted(tags & set(self.profile.tags))) or '-'}"
                votes.append(PreferenceVote(self.name, item.id, score, reason))
        return votes


class ItemAgent:
    """A candidate item that argues for its own relevance."""

    def __init__(self, item: Item) -> None:
        self.item = item
        self.name = f"ItemAgent:{item.id}"

    def vote(self, query_tags: Iterable[str], user_tags: Dict[str, float]) -> PreferenceVote:
        tags = set(self.item.features.get("tags", []))
        q_tags = set(query_tags)
        query_fit = len(tags & q_tags) / max(1, len(q_tags))
        user_fit = sum(user_tags.get(tag, 0.0) for tag in tags) / max(1, len(tags))
        freshness = float(self.item.features.get("freshness", 0.0))
        score = 0.45 * query_fit + 0.35 * user_fit + 0.20 * freshness
        reason = f"item_tags={','.join(sorted(tags))}, query_fit={query_fit:.2f}, freshness={freshness:.2f}"
        return PreferenceVote(self.name, self.item.id, score, reason)


class CollaborationOrchestrator:
    """Dynamic recruiter and vote aggregator for user/item agents."""

    def __init__(self, neighbor_profiles: Sequence[AgentProfile] | None = None) -> None:
        self.neighbor_profiles = list(neighbor_profiles or [])

    def run(
        self,
        items: Sequence[Item],
        user_tags: Dict[str, float],
        query: str,
        top_users: int = 3,
        top_items: int = 6,
    ) -> CollaborationReport:
        query_tags = _query_tags(query, items)
        users = self._recruit_users(user_tags, query_tags, limit=top_users)
        candidates = _top_non_ad_items(items, limit=top_items)

        votes: List[PreferenceVote] = []
        for user in users:
            votes.extend(SimilarUserAgent(user).vote(candidates, query_tags))
        for item in candidates:
            votes.append(ItemAgent(item).vote(query_tags, user_tags))

        item_scores = _aggregate_votes(votes)
        thought = (
            f"recruited {len(users)} user agents and {len(candidates)} item agents; "
            f"query_tags={sorted(query_tags)}; votes={len(votes)}"
        )
        return CollaborationReport(
            recruited_users=[u.id for u in users],
            recruited_items=[i.id for i in candidates],
            votes=votes,
            item_scores=item_scores,
            thought=thought,
        )

    def _recruit_users(
        self,
        user_tags: Dict[str, float],
        query_tags: set[str],
        limit: int,
    ) -> List[AgentProfile]:
        scored: List[tuple[float, AgentProfile]] = []
        active_tags = set(user_tags) | query_tags
        for profile in self.neighbor_profiles:
            tags = set(profile.tags)
            overlap = tags & active_tags
            if not overlap:
                continue
            score = sum(profile.tags.get(tag, 0.0) + user_tags.get(tag, 0.0) for tag in overlap)
            score += 0.2 * len(overlap)
            scored.append((score, profile))
        scored.sort(key=lambda x: -x[0])
        return [profile for _, profile in scored[:limit]]


def build_neighbor_profiles(corpus: Sequence[Dict[str, Any]]) -> List[AgentProfile]:
    """Build deterministic neighbor users from item tags.

    Industrial systems would recruit real similar users from a CF/ANN index.
    Here we synthesize compact profiles so the collaboration flow is runnable
    immediately after clone.
    """

    by_tag: Dict[str, List[str]] = {}
    for item in corpus:
        for tag in item.get("tags", []):
            by_tag.setdefault(tag, []).append(item["id"])

    profiles: List[AgentProfile] = []
    for idx, (tag, history) in enumerate(sorted(by_tag.items()), start=1):
        co_tags = Counter()
        for item in corpus:
            if item["id"] in history:
                for other in item.get("tags", []):
                    co_tags[other] += 1
        tag_weights = {k: round(v / max(1, len(history)), 3) for k, v in co_tags.items()}
        profiles.append(AgentProfile(id=f"neighbor_{idx}_{tag}", tags=tag_weights, history=history[:5]))
    return profiles


def apply_collaboration_scores(
    items: Sequence[Item],
    report: CollaborationReport,
    alpha: float = 0.35,
) -> List[Item]:
    """Blend collaborative votes back into item scores."""

    out: List[Item] = []
    for item in items:
        if item.features.get("is_ad"):
            out.append(item)
            continue
        boost = report.item_scores.get(item.id, 0.0)
        if boost:
            item.score = item.score * (1.0 - alpha) + boost * alpha
            item.features["collab_score"] = round(boost, 4)
        out.append(item)
    out.sort(key=lambda x: (x.features.get("is_ad", False), -x.score))
    return out


def _query_tags(query: str, items: Sequence[Item]) -> set[str]:
    query_lower = query.lower()
    known_tags = {tag for item in items for tag in item.features.get("tags", [])}
    hits = {tag for tag in known_tags if tag.lower() in query_lower}
    if hits:
        return hits
    # Fall back to the dominant tags in current candidates.
    counter = Counter(tag for item in items for tag in item.features.get("tags", []))
    return {tag for tag, _ in counter.most_common(2)}


def _top_non_ad_items(items: Sequence[Item], limit: int) -> List[Item]:
    candidates = [item for item in items if not item.features.get("is_ad")]
    candidates.sort(key=lambda item: -item.score)
    return candidates[:limit]


def _aggregate_votes(votes: Sequence[PreferenceVote]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for vote in votes:
        grouped.setdefault(vote.item_id, []).append(vote.score)
    return {item_id: sum(scores) / len(scores) for item_id, scores in grouped.items()}
