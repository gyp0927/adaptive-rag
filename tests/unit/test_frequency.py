"""Unit tests for frequency tracking."""

import math
from datetime import datetime, timedelta

from adaptive_rag.frequency.decay import DecayEngine


class TestDecayEngine:
    """Test decay engine calculations."""

    def test_apply_decay_no_access(self):
        """Score should remain unchanged if never accessed."""
        engine = DecayEngine()
        score = engine.apply_decay(
            base_score=1.0,
            last_accessed=None,
            access_count=0,
        )
        assert score == 1.0

    def test_apply_decay_recent_access(self):
        """Recent access should have minimal decay."""
        engine = DecayEngine()
        now = datetime.utcnow()
        score = engine.apply_decay(
            base_score=1.0,
            last_accessed=now,
            access_count=5,
        )
        # Just created, almost no decay
        assert score > 0.99

    def test_apply_decay_old_access(self):
        """Old access should decay significantly."""
        engine = DecayEngine()
        old_time = datetime.utcnow() - timedelta(hours=48)
        score = engine.apply_decay(
            base_score=1.0,
            last_accessed=old_time,
            access_count=1,
        )
        # After 2 half-lives, should be around 0.25
        assert score < 0.5
        # But never below minimum
        assert score >= math.log1p(1) / 6.0

    def test_compute_score_new_chunk(self):
        """New chunk with no accesses should have low score."""
        engine = DecayEngine()
        now = datetime.utcnow()
        score = engine.compute_score(
            access_count=0,
            last_accessed=None,
            created_at=now,
            cluster_score=0.0,
        )
        assert score >= 0.0
        assert score < 0.5

    def test_compute_score_popular_chunk(self):
        """Popular chunk should have high score."""
        engine = DecayEngine()
        now = datetime.utcnow()
        score = engine.compute_score(
            access_count=100,
            last_accessed=now,
            created_at=now,
            cluster_score=50.0,
        )
        assert score > 0.5
        assert score <= 1.0
