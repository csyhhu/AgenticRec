"""MovieLens-1M adapter for AgenticRec.

Converts the MovieLens-1M dataset into AgenticRec-native formats:

- corpus: each movie becomes an Item with id / title / tags / ctr_prior / freshness / hot.
- scenarios: classic / cold_start / evolving_interest, each with ~200 BenchTasks.
- neighbor_profiles: built from genre co-occurrence patterns.

Data source:
    https://grouplens.org/datasets/movielens/1m/

Dependencies (install before use):
    pip install pandas
"""

from __future__ import annotations

import os
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..bench import BenchTask, Scenario
from ..collab import AgentProfile
from .base import DatasetAdapter


# MovieLens-1M genre list (18 genres, pipe-separated in the raw data)
ML_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Mapping from ML genre to a Chinese-friendly label (optional; English is default)
GENRE_LABELS: Dict[str, str] = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Children's": "Children",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Documentary": "Documentary",
    "Drama": "Drama",
    "Fantasy": "Fantasy",
    "Film-Noir": "Film-Noir",
    "Horror": "Horror",
    "Musical": "Musical",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Sci-Fi": "Sci-Fi",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western",
}

# Natural-language templates for generating queries from genre preferences
QUERY_TEMPLATES: List[str] = [
    "I want {genres} movies",
    "Recommend me some {genres} films",
    "Looking for {genres}",
    "Show me {genres} movies",
    "I'm in the mood for {genres}",
    "Find {genres} films for me",
    "Give me {genres} recommendations",
]


class MovieLensAdapter(DatasetAdapter):
    """Adapter for the MovieLens-1M dataset.

    Parameters
    ----------
    raw_path : str | None
        Path to the directory containing ratings.dat, movies.dat, users.dat.
        If None, defaults to ``~/.agentic_rec/ml-1m/``.

    min_rating : float
        Ratings >= this value are considered "relevant" for ground-truth.
        Default 4.0.

    min_user_interactions : int
        Users with fewer than this many ratings are excluded from evaluation.
        Default 10.

    seed : int
        Random seed for train/test split reproducibility. Default 42.
    """

    name = "movielens-1m"

    def __init__(
        self,
        raw_path: Optional[str] = None,
        min_rating: float = 4.0,
        min_user_interactions: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.raw_path = raw_path or os.path.join(
            os.path.expanduser("~"), ".agentic_rec", "ml-1m"
        )
        self.min_rating = min_rating
        self.min_user_interactions = min_user_interactions
        self.seed = seed

        # Internal caches
        self._movies_df: Optional[pd.DataFrame] = None
        self._ratings_df: Optional[pd.DataFrame] = None
        self._corpus: Optional[List[Dict[str, Any]]] = None
        self._corpus_index: Optional[Dict[str, Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _ensure_data(self) -> None:
        """Lazy-load the raw MovieLens files into pandas DataFrames."""
        if self._movies_df is not None and self._ratings_df is not None:
            return

        movies_path = os.path.join(self.raw_path, "movies.dat")
        ratings_path = os.path.join(self.raw_path, "ratings.dat")

        if not os.path.exists(movies_path):
            raise FileNotFoundError(
                f"MovieLens-1M movies.dat not found at {movies_path}. "
                f"Download from https://grouplens.org/datasets/movielens/1m/ "
                f"and extract to {self.raw_path}"
            )
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"MovieLens-1M ratings.dat not found at {ratings_path}."
            )

        self._movies_df = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        self._ratings_df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def load_corpus(self) -> List[Dict[str, Any]]:
        if self._corpus is not None:
            return self._corpus

        self._ensure_data()
        movies = self._movies_df
        ratings = self._ratings_df

        # Compute per-movie statistics
        movie_stats = ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            timestamp_max=("timestamp", "max"),
            timestamp_min=("timestamp", "min"),
        ).reset_index()

        # Merge with movie metadata
        merged = movies.merge(movie_stats, on="movieId", how="left").fillna(0)

        # Normalise fields
        avg_ratings = merged["avg_rating"].tolist()
        rating_counts = merged["rating_count"].tolist()
        time_spans = (merged["timestamp_max"] - merged["timestamp_min"]).tolist()

        ctr_priors = self._normalise(avg_ratings)
        hot_values = [math.log1p(c) for c in rating_counts]  # log-scale hot
        hot_values = self._normalise(hot_values)
        freshness_values = self._normalise(time_spans)
        # Invert freshness: larger time span = older = less fresh
        freshness_values = [1.0 - v for v in freshness_values]

        corpus: List[Dict[str, Any]] = []
        for i, row in merged.iterrows():
            genres = [g.strip() for g in str(row["genres"]).split("|") if g.strip()]
            corpus.append({
                "id": f"ml_{int(row['movieId'])}",
                "title": str(row["title"]),
                "tags": genres,
                "author": genres[0] if genres else "Unknown",  # first genre as proxy
                "ctr_prior": round(ctr_priors[i], 4),
                "freshness": round(freshness_values[i], 4),
                "hot": round(hot_values[i], 4),
            })

        self._corpus = corpus
        self._corpus_index = {item["id"]: item for item in corpus}
        return corpus

    @property
    def corpus_index(self) -> Dict[str, Dict[str, Any]]:
        if self._corpus_index is None:
            self.load_corpus()
        return self._corpus_index  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Neighbor profiles
    # ------------------------------------------------------------------

    def build_neighbor_profiles(self) -> List[AgentProfile]:
        corpus = self.load_corpus()

        # Group items by each tag, then compute co-tag weights
        by_tag: Dict[str, List[str]] = {}
        for item in corpus:
            for tag in item["tags"]:
                by_tag.setdefault(tag, []).append(item["id"])

        profiles: List[AgentProfile] = []
        for tag, history in sorted(by_tag.items()):
            co_tags = Counter()
            for item in corpus:
                if item["id"] in history:
                    for other in item["tags"]:
                        co_tags[other] += 1
            tag_weights = {
                k: round(v / max(1, len(history)), 4)
                for k, v in co_tags.items()
            }
            profiles.append(
                AgentProfile(
                    id=f"neighbor_{tag.lower()}",
                    tags=tag_weights,
                    history=history[:10],  # keep top 10 as history
                )
            )
        return profiles

    # ------------------------------------------------------------------
    # User training data (for ItemKNN baseline)
    # ------------------------------------------------------------------

    def build_user_train_map(
        self,
        max_users: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Build a user→item rating map from MovieLens training splits.

        Uses the same train/test split as ``load_scenarios`` (80/20 by timestamp),
        so there is no data leakage between training data and evaluation tasks.

        Parameters
        ----------
        max_users : int | None
            Cap the number of users for faster ItemKNN computation.

        Returns
        -------
        user_train_map : Dict[str, Dict[str, float]]
            ``{user_id: {item_id: rating, ...}, ...}`` for training items only.
        """
        self._ensure_data()
        self.load_corpus()  # ensure _corpus_index is built

        # Build per-user rating history
        user_ratings: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
        for _, row in self._ratings_df.iterrows():
            uid = int(row["userId"])
            mid = int(row["movieId"])
            user_ratings[uid].append((mid, float(row["rating"]), int(row["timestamp"])))

        for uid in user_ratings:
            user_ratings[uid].sort(key=lambda x: x[2])

        user_train_map: Dict[str, Dict[str, float]] = {}
        users = list(user_ratings.keys())
        if max_users is not None:
            import random
            rng = random.Random(self.seed)
            users = rng.sample(users, min(max_users, len(users)))

        for uid in users:
            hist = user_ratings[uid]
            if len(hist) < 2:
                continue
            split = max(1, int(len(hist) * 0.8))
            train = hist[:split]
            user_train_map[f"ml_u_{uid}"] = {
                f"ml_{mid}": rating for mid, rating, _ in train
            }

        return user_train_map

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------

    def load_scenarios(self) -> List[Scenario]:
        self._ensure_data()
        corpus = self.load_corpus()

        # Build user-item interaction map
        user_ratings: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
        # userId -> [(movieId, rating, timestamp), ...]
        for _, row in self._ratings_df.iterrows():
            uid = int(row["userId"])
            mid = int(row["movieId"])
            user_ratings[uid].append((mid, float(row["rating"]), int(row["timestamp"])))

        # Sort each user's ratings by timestamp
        for uid in user_ratings:
            user_ratings[uid].sort(key=lambda x: x[2])

        # Filter users with enough interactions for classic/evolving scenarios
        qualified_users = {
            uid: hist
            for uid, hist in user_ratings.items()
            if len(hist) >= self.min_user_interactions
        }

        # Cold-start users: few interactions but at least enough for a train/test split
        # MovieLens-1M has min 20 ratings per user, so we use "low interaction" as proxy
        cold_users: List[int] = [
            uid for uid, hist in user_ratings.items()
            if self.min_user_interactions <= len(hist) <= 30
        ]

        # Split qualified users into classic and evolving
        classic_users: List[int] = []
        evolving_users: List[int] = []

        for uid, hist in qualified_users.items():
            n = len(hist)
            # Check for evolving interest: compare first half vs second half genres
            mid = n // 2
            first_genres = self._user_genre_set(uid, hist[:mid])
            second_genres = self._user_genre_set(uid, hist[mid:])
            overlap = first_genres & second_genres
            if overlap and len(overlap) < max(1, len(first_genres)) * 0.5:
                evolving_users.append(uid)
            else:
                classic_users.append(uid)

        # Build scenarios
        rng = __import__("random")
        rng.seed(self.seed)

        scenarios = [
            self._build_scenario("classic", classic_users, corpus, "stable profile + explicit query", rng),
            self._build_scenario("cold_start", cold_users, corpus, "few interactions; rely on query + hot fallback", rng),
            self._build_scenario("evolving_interest", evolving_users, corpus, "profile says one thing, query reveals fresh intent", rng),
        ]
        return scenarios

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _user_genre_set(
        self,
        uid: int,
        history: List[Tuple[int, float, int]],
    ) -> set:
        """Collect all genres from a user's interaction history."""
        genres: set = set()
        for mid, rating, _ in history:
            if rating >= self.min_rating:
                item = self.corpus_index.get(f"ml_{mid}")
                if item:
                    genres.update(item["tags"])
        return genres

    def _build_scenario(
        self,
        name: str,
        user_pool: List[int],
        corpus: List[Dict[str, Any]],
        description: str,
        rng,
        max_tasks: int = 200,
    ) -> Scenario:
        """Build a Scenario from a pool of user IDs."""
        # Shuffle and sample
        sampled = rng.sample(user_pool, min(len(user_pool), max_tasks))

        tasks: List[BenchTask] = []
        for uid in sampled:
            hist = self._ratings_df[
                self._ratings_df["userId"] == uid
            ].sort_values("timestamp")

            if len(hist) < 2:
                continue

            # Split: first 80% for profile, last 20% for ground-truth
            split = max(1, int(len(hist) * 0.8))
            train = hist.iloc[:split]
            test = hist.iloc[split:]

            # Build profile tags from training ratings
            train_ratings: Dict[str, float] = {}
            for _, row in train.iterrows():
                train_ratings[f"ml_{int(row['movieId'])}"] = float(row["rating"])

            profile_tags = self._build_user_profile_tags(train_ratings, self.corpus_index)

            # Ground-truth relevant items from test set
            relevant_ids: List[str] = []
            for _, row in test.iterrows():
                if float(row["rating"]) >= self.min_rating:
                    relevant_ids.append(f"ml_{int(row['movieId'])}")

            if not relevant_ids:
                continue

            # Generate a natural-language query from top profile genres
            query = self._generate_query(profile_tags)

            tasks.append(BenchTask(
                user_id=f"ml_u_{uid}",
                query=query,
                scene="feed_home",
                relevant_ids=relevant_ids,
                profile_tags=profile_tags,
            ))

        return Scenario(name=name, description=description, tasks=tasks)

    def _generate_query(self, profile_tags: Dict[str, float]) -> str:
        """Generate a natural-language query from profile tag preferences."""
        if not profile_tags:
            return "popular movies"

        # Pick top 1-2 genres
        sorted_tags = sorted(profile_tags.items(), key=lambda x: -x[1])
        top_genres = [tag for tag, _ in sorted_tags[:2]]

        import random
        rng = random.Random(hash(tuple(sorted_tags)) % (2**31))
        template = rng.choice(QUERY_TEMPLATES)
        genres_str = " and ".join(top_genres)
        return template.format(genres=genres_str)

    # ------------------------------------------------------------------
    # Validation / self-check
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """Run self-validation checks and return a summary dict.

        Checks performed:
        1. Corpus: non-empty, all items have required fields.
        2. Neighbor profiles: non-empty, each profile has valid tags/history.
        3. Scenarios: at least one task per scenario, each task has query +
           relevant_ids + profile_tags; no empty relevant_ids.
        4. User train map: non-empty, each user has at least one item.

        Returns
        -------
        report : dict
            ``{"status": "ok"|"warn", "checks": {...}, "errors": [...]}``
        """
        report: Dict[str, Any] = {"status": "ok", "checks": {}, "errors": []}

        # ---- 1. Corpus ----
        corpus = self.load_corpus()
        report["checks"]["corpus_count"] = len(corpus)
        if not corpus:
            report["status"] = "warn"
            report["errors"].append("Corpus is empty")
        else:
            missing = []
            for item in corpus:
                for field in ("id", "title", "tags", "ctr_prior", "freshness", "hot"):
                    if field not in item:
                        missing.append(f"item {item.get('id', '?')} missing field '{field}'")
            if missing:
                report["status"] = "warn"
                report["errors"].extend(missing[:10])
            report["checks"]["corpus_fields_ok"] = len(missing) == 0

            # Tag distribution
            tag_counter = Counter()
            for item in corpus:
                for t in item.get("tags", []):
                    tag_counter[t] += 1
            report["checks"]["tag_distribution"] = dict(tag_counter.most_common(10))

        # ---- 2. Neighbor profiles ----
        profiles = self.build_neighbor_profiles()
        report["checks"]["profile_count"] = len(profiles)
        if not profiles:
            report["status"] = "warn"
            report["errors"].append("Neighbor profiles are empty")
        else:
            sample = []
            for p in profiles[:3]:
                sample.append({
                    "id": p.id,
                    "tag_count": len(p.tags),
                    "history_len": len(p.history),
                })
            report["checks"]["profile_sample"] = sample

        # ---- 3. Scenarios ----
        scenarios = self.load_scenarios()
        scenario_summary = {}
        total_tasks = 0
        for s in scenarios:
            n = len(s.tasks)
            total_tasks += n
            scenario_summary[s.name] = {"task_count": n, "description": s.description}

            # Spot-check: every task should have query, relevant_ids, profile_tags
            empty_rel = sum(1 for t in s.tasks if not t.relevant_ids)
            empty_q = sum(1 for t in s.tasks if not t.query)
            scenario_summary[s.name]["empty_relevant"] = empty_rel
            scenario_summary[s.name]["empty_query"] = empty_q
            if empty_rel:
                report["status"] = "warn"
                report["errors"].append(
                    f"Scenario '{s.name}': {empty_rel}/{n} tasks have empty relevant_ids"
                )
            if empty_q:
                report["status"] = "warn"
                report["errors"].append(
                    f"Scenario '{s.name}': {empty_q}/{n} tasks have empty query"
                )

        report["checks"]["scenarios"] = scenario_summary
        report["checks"]["total_tasks"] = total_tasks
        if total_tasks == 0:
            report["status"] = "warn"
            report["errors"].append("No tasks generated across all scenarios")

        # ---- 4. User train map ----
        user_train_map = self.build_user_train_map()
        report["checks"]["user_train_count"] = len(user_train_map)
        if not user_train_map:
            report["status"] = "warn"
            report["errors"].append("User train map is empty")
        else:
            # Check min/max items per user
            item_counts = [len(v) for v in user_train_map.values()]
            report["checks"]["user_train_items"] = {
                "min": min(item_counts),
                "max": max(item_counts),
                "avg": round(sum(item_counts) / len(item_counts), 1),
            }

        return report

    def print_validation_report(self) -> None:
        """Print a human-readable validation report to stdout."""
        report = self.validate()

        print("=" * 60)
        print(f"  MovieLensAdapter Validation  [status: {report['status']}]")
        print(f"  Data path: {self.raw_path}")
        print("=" * 60)

        # Corpus
        print(f"\n--- Corpus ---")
        print(f"  Items: {report['checks']['corpus_count']}")
        print(f"  Fields valid: {report['checks']['corpus_fields_ok']}")
        if "tag_distribution" in report["checks"]:
            tags = report["checks"]["tag_distribution"]
            print(f"  Top tags: {dict(list(tags.items())[:5])}")

        # Profiles
        print(f"\n--- Neighbor Profiles ---")
        print(f"  Count: {report['checks']['profile_count']}")
        for p in report["checks"].get("profile_sample", []):
            print(f"  {p['id']}: {p['tag_count']} tags, {p['history_len']} history")

        # Scenarios
        print(f"\n--- Scenarios ---")
        for name, info in report["checks"]["scenarios"].items():
            print(f"  {name}: {info['task_count']} tasks | {info['description']}")
            if info.get("empty_relevant"):
                print(f"    WARNING: {info['empty_relevant']} tasks with empty relevant_ids")
            if info.get("empty_query"):
                print(f"    WARNING: {info['empty_query']} tasks with empty query")
        print(f"  Total tasks: {report['checks']['total_tasks']}")

        # User train map
        print(f"\n--- User Train Map ---")
        print(f"  Users: {report['checks']['user_train_count']}")
        uti = report["checks"].get("user_train_items")
        if uti:
            print(f"  Items/user: min={uti['min']} max={uti['max']} avg={uti['avg']}")

        # Errors
        if report["errors"]:
            print(f"\n--- Issues ({len(report['errors'])}) ---")
            for e in report["errors"]:
                print(f"  [!] {e}")

        print(f"\nValidation {'PASSED' if report['status'] == 'ok' else 'COMPLETED WITH ISSUES'}")
