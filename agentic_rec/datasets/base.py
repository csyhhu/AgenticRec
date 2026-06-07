"""Base protocol for dataset adapters.

A DatasetAdapter is responsible for converting a public recommendation dataset
into three standardised artefacts:

1. corpus: List[Dict] — the item catalogue in AgenticRec format.
2. scenarios: List[Scenario] — evaluation scenarios with BenchTasks.
3. neighbor_profiles: List[AgentProfile] — profiles for the collaboration layer.

Any new dataset adapter should implement these three methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..collab import AgentProfile
from ..bench import Scenario


@dataclass
class DatasetAdapter(ABC):
    """Abstract base for dataset-to-AgenticRec conversion.

    Subclasses must implement load_corpus, load_scenarios, and
    build_neighbor_profiles.
    """

    name: str = "base"
    raw_path: str | None = None  # path to raw data files

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_corpus(self) -> List[Dict[str, Any]]:
        """Return the item catalogue.

        Each dict MUST contain at least: id, title, tags.
        Recommended: author, ctr_prior, freshness, hot.
        """

    @abstractmethod
    def load_scenarios(self) -> List[Scenario]:
        """Return evaluation scenarios populated with BenchTasks."""

    @abstractmethod
    def build_neighbor_profiles(self) -> List[AgentProfile]:
        """Build neighbour user profiles for the collaboration layer."""

    # ------------------------------------------------------------------
    # Convenience helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(values: List[float], floor: float = 0.0, ceil: float = 1.0) -> List[float]:
        """Min-max normalise a list of floats to [floor, ceil]."""
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi == lo:
            return [floor] * len(values)
        return [floor + (v - lo) / (hi - lo) * (ceil - floor) for v in values]

    @staticmethod
    def _build_user_profile_tags(
        user_ratings: Dict[str, float],  # item_id -> rating
        corpus_index: Dict[str, Dict[str, Any]],
        min_rating: float = 4.0,
    ) -> Dict[str, float]:
        """Build a user profile tag dict from their ratings.

        For each highly-rated item, accumulate its tags weighted by the rating.
        Then normalise so the weights sum to 1.
        """
        accum: Dict[str, float] = {}
        for item_id, rating in user_ratings.items():
            if rating < min_rating:
                continue
            item = corpus_index.get(item_id)
            if not item:
                continue
            tags = item.get("tags", [])
            weight = rating / 5.0  # normalise to [0, 1]
            for tag in tags:
                accum[tag] = accum.get(tag, 0.0) + weight
        if not accum:
            return {}
        total = sum(accum.values())
        return {tag: round(val / total, 4) for tag, val in accum.items()}
