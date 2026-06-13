from unittest.mock import AsyncMock, MagicMock

import pytest

from hot_and_cold_memory.retrieval.retriever import UnifiedRetriever


@pytest.mark.asyncio
async def test_retrieve_uses_profile_augmentation():
    mock_router = MagicMock()
    mock_router.route = AsyncMock(return_value=MagicMock(chunks=[], routing_strategy=MagicMock(value="hot_only"), hot_results_count=0, cold_results_count=0, total_latency_ms=1.0, topic_frequency=0.0))

    retriever = UnifiedRetriever(
        hot_tier=MagicMock(),
        cold_tier=MagicMock(),
        frequency_tracker=MagicMock(),
    )
    retriever.router = mock_router

    augmenter = MagicMock()
    augmenter.rewrite_query = AsyncMock(return_value="expanded query")
    augmenter.rerank = AsyncMock(return_value=[])
    retriever.profile_augmenter = augmenter

    await retriever.query("test", use_profile=True)
    augmenter.rewrite_query.assert_awaited_once()
    augmenter.rerank.assert_awaited_once()
