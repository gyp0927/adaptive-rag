"""Exponential time decay for frequency scores.

Recent accesses weighted higher; old accesses fade over time.
"""

import math
from datetime import datetime

from adaptive_rag.core.config import get_settings


class DecayEngine:
    """Implements exponential time decay for frequency scores."""

    def __init__(self) -> None:
        settings = get_settings()
        self.half_life_seconds = settings.DECAY_HALF_LIFE_HOURS * 3600
        self.decay_constant = math.log(2) / self.half_life_seconds

    def apply_decay(
        self,
        base_score: float,
        last_accessed: datetime | None,
        access_count: int,
    ) -> float:
        """Apply time decay to a frequency score.

        Args:
            base_score: The stored frequency score.
            last_accessed: When the chunk was last accessed.
            access_count: Total number of accesses.

        Returns:
            Decayed score.
        """
        if last_accessed is None:
            return base_score

        elapsed = (datetime.utcnow() - last_accessed).total_seconds()
        decay_factor = math.exp(-self.decay_constant * elapsed)

        # Score decays but never below a minimum based on total accesses
        min_score = math.log1p(access_count) / 6.0

        return max(base_score * decay_factor, min_score)

    def compute_score(
        self,
        access_count: int,
        last_accessed: datetime | None,
        created_at: datetime,
        cluster_score: float,
    ) -> float:
        """Compute composite frequency score.

        Combines:
        - Individual chunk access count
        - Time since last access (decay)
        - Topic cluster popularity

        Args:
            access_count: Number of times chunk was accessed.
            last_accessed: Last access timestamp.
            created_at: Chunk creation timestamp.
            cluster_score: Topic cluster frequency score.

        Returns:
            Normalized frequency score in [0, 1].
        """
        # Base score from access count
        access_component = math.log1p(access_count)

        # Recency component (higher = more recent)
        if last_accessed:
            age_seconds = (datetime.utcnow() - last_accessed).total_seconds()
            recency = math.exp(-self.decay_constant * age_seconds)
        else:
            recency = 0.0

        # Cluster popularity component
        cluster_component = math.log1p(cluster_score)

        # Weighted combination
        score = (
            0.4 * access_component +
            0.3 * recency * 10 +
            0.3 * cluster_component
        )

        # Normalize to [0, 1]
        return min(score / 10.0, 1.0)
