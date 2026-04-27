"""On-demand decompression engine for cold tier."""

import math
from dataclasses import dataclass

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.exceptions import DecompressionError
from adaptive_rag.core.llm_client import LLMClient
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.embedder import Embedder

logger = get_logger(__name__)


@dataclass
class DecompressionResult:
    """Result of a validated decompression."""

    text: str
    relevance: float
    flagged_for_review: bool


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DecompressionEngine:
    """On-demand decompression that expands cold summaries.

    Uses the compressed summary + LLM to generate a contextually
    appropriate expansion while maintaining factual accuracy.
    """

    DECOMPRESSION_PROMPT = """You are a knowledge expansion engine. Expand the following compressed summary into a detailed, informative response.

Compressed summary:
{summary}

Instructions:
1. Expand the summary into a comprehensive response
2. Include all key entities and facts from the summary
3. Add relevant context and explanations where appropriate
4. Maintain factual accuracy - do not hallucinate information not in the summary
5. Write in a clear, professional tone

Expanded response:
"""

    # Below this query<->expansion similarity, the expansion is considered low quality
    DEFAULT_RELEVANCE_THRESHOLD = 0.6

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.settings = get_settings()
        self.client = LLMClient()
        self.model = self.settings.DECOMPRESSION_MODEL
        self.embedder = embedder
        self._flagged_chunk_ids: set[str] = set()

    async def decompress(self, summary: str) -> str:
        """Decompress a summary back to full detail.

        Args:
            summary: Compressed summary text.

        Returns:
            Expanded/detailed text.
        """
        prompt = self.DECOMPRESSION_PROMPT.format(summary=summary)

        try:
            response = await self.client.complete(
                prompt=prompt,
                model=self.model,
                max_tokens=self.settings.LLM_MAX_TOKENS,
                temperature=0.3,
            )

            logger.debug(
                "chunk_decompressed",
                summary_len=len(summary),
                expanded_len=len(response),
            )

            return response

        except Exception as e:
            logger.error("decompression_failed", error=str(e))
            raise DecompressionError(f"Failed to decompress: {e}") from e

    async def decompress_and_validate(
        self,
        compressed: str,
        query: str,
        chunk_id: str | None = None,
        threshold: float | None = None,
    ) -> DecompressionResult:
        """Decompress and verify the expansion stays on-topic for the query.

        If the expansion drifts away from the query (cosine similarity below
        threshold), the original compressed text is returned instead and the
        chunk is flagged for review. This prevents low-quality expansions from
        being surfaced to the user.

        Args:
            compressed: Compressed summary text.
            query: Original query text used for relevance check.
            chunk_id: Optional chunk ID to flag if quality is low.
            threshold: Minimum cosine similarity (default 0.6).

        Returns:
            DecompressionResult with text, relevance score, and flag status.
        """
        threshold = threshold if threshold is not None else self.DEFAULT_RELEVANCE_THRESHOLD
        embedder = self.embedder or Embedder()

        decompressed = await self.decompress(compressed)

        try:
            decompressed_emb, query_emb = await embedder.embed_batch([decompressed, query])
        except Exception as e:
            logger.warning("decompression_validation_embed_failed", error=str(e))
            return DecompressionResult(
                text=decompressed, relevance=1.0, flagged_for_review=False
            )

        relevance = _cosine_similarity(decompressed_emb, query_emb)

        if relevance < threshold:
            if chunk_id:
                self.flag_for_review(chunk_id)
            logger.warning(
                "decompression_low_quality",
                chunk_id=chunk_id,
                relevance=relevance,
                threshold=threshold,
            )
            # Return the safer compressed version when expansion drifts
            return DecompressionResult(
                text=compressed,
                relevance=relevance,
                flagged_for_review=True,
            )

        return DecompressionResult(
            text=decompressed, relevance=relevance, flagged_for_review=False
        )

    def flag_for_review(self, chunk_id: str) -> None:
        """Mark a chunk's decompression as needing human review."""
        self._flagged_chunk_ids.add(str(chunk_id))

    @property
    def flagged_chunk_ids(self) -> set[str]:
        """Chunks whose decompressions were rejected on quality grounds."""
        return set(self._flagged_chunk_ids)
