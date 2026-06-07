"""AgenticRec public dataset adapters.

Each adapter converts a well-known public recommendation dataset into the
AgenticRec-native format (corpus + scenarios + neighbor profiles) so that the
benchmark pipeline can run without any code changes.
"""

from .base import DatasetAdapter
from .movielens import MovieLensAdapter

__all__ = [
    "DatasetAdapter",
    "MovieLensAdapter",
]
