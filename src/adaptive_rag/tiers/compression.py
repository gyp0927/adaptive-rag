"""LLM-based compression engine for cold tier."""

import asyncio
import json
import re
from dataclasses import dataclass
import uuid

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.exceptions import CompressionError
from adaptive_rag.core.llm_client import LLMClient
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.chunker import Chunk

logger = get_logger(__name__)


@dataclass
class CompressedChunk:
    """A compressed chunk with preserved key information."""

    chunk_id: uuid.UUID
    summary_text: str
    key_entities: list[str]
    key_facts: list[str]
    compression_ratio: float


class CompressionEngine:
    """LLM-based compression that preserves semantic meaning.

    The single-chunk prompt enforces strict entity / number / date preservation.
    The grouped prompt amortizes a single LLM call over many chunks to cut cost
    by ~10x for migration workloads where per-chunk fidelity is less critical
    than throughput.
    """

    COMPRESSION_PROMPT = """You are a knowledge compression engine. Compress the text below while preserving information that retrieval depends on.

Strict requirements:
1. PRESERVE every number, date, percentage, currency amount, and unit
2. PRESERVE every proper noun: person names, organizations, locations, products, code/IDs
3. PRESERVE every quoted phrase verbatim
4. PRESERVE the core conclusion and any causal relationships
5. Drop only filler, repetition, hedging, and rhetorical scaffolding
6. Compress to roughly {target_ratio}% of the original length
7. Output language MUST match the original text

Original text:
{text}

Output as JSON:
{{
    "summary": "compressed text",
    "key_entities": ["entity1", "entity2"],
    "key_facts": ["fact1", "fact2"]
}}
"""

    GROUP_COMPRESSION_PROMPT = """You are a knowledge compression engine. Compress {count} text segments below, applying the SAME strict preservation rules to each:

1. Keep every number, date, percentage, currency amount, unit, proper noun, and quoted phrase
2. Keep the core conclusion and causal relationships
3. Drop only filler / repetition / hedging
4. Compress each to roughly {target_ratio}% of its original length
5. Output language MUST match the segment's original language

Segments (each prefixed with [index]):
{combined_text}

Output a JSON array with one object per segment, in the same order:
[
  {{"index": 0, "summary": "...", "key_entities": [...], "key_facts": [...]}},
  {{"index": 1, "summary": "...", "key_entities": [...], "key_facts": [...]}}
]
"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = LLMClient()
        self.model = self.settings.COMPRESSION_MODEL

    async def compress(self, chunk: Chunk) -> CompressedChunk:
        """Compress a single chunk.

        Args:
            chunk: Chunk to compress.

        Returns:
            Compressed chunk.
        """
        target_ratio = int(self.settings.COLD_TIER_COMPRESSION_RATIO * 100)

        prompt = self.COMPRESSION_PROMPT.format(
            text=chunk.text,
            target_ratio=target_ratio,
        )

        try:
            # Use JSON response format for OpenAI, plain text for Anthropic
            if self.client._is_anthropic_format():
                response_text = await self.client.complete(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=self.settings.COMPRESSION_MAX_TOKENS,
                    temperature=0.0,
                )
                result = self._parse_json_response(response_text)
            else:
                response_text = await self.client.complete(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=self.settings.COMPRESSION_MAX_TOKENS,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response_text)

            summary = result.get("summary", "")

            compression_ratio = len(summary) / max(len(chunk.text), 1)

            logger.debug(
                "chunk_compressed",
                chunk_id=str(chunk.chunk_id),
                original_len=len(chunk.text),
                compressed_len=len(summary),
                ratio=compression_ratio,
            )

            return CompressedChunk(
                chunk_id=chunk.chunk_id,
                summary_text=summary,
                key_entities=result.get("key_entities", []),
                key_facts=result.get("key_facts", []),
                compression_ratio=compression_ratio,
            )

        except Exception as e:
            logger.error("compression_failed", chunk_id=str(chunk.chunk_id), error=str(e))
            raise CompressionError(f"Failed to compress chunk {chunk.chunk_id}: {e}") from e

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response text (for non-OpenAI models)."""
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Fallback: wrap entire response as summary
        return {"summary": text, "key_entities": [], "key_facts": []}

    def _parse_json_array_response(self, text: str) -> list[dict]:
        """Extract JSON array from LLM response text."""
        # Strip code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "results" in parsed:
                return parsed["results"]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    async def compress_group(self, chunks: list[Chunk]) -> list[CompressedChunk]:
        """Compress a group of chunks in a single LLM call.

        Combines multiple chunks into one prompt to cut LLM cost by ~10x.
        Falls back to per-chunk compression if the grouped response fails to parse
        or returns the wrong number of summaries.

        Args:
            chunks: Chunks to compress in one call.

        Returns:
            List of compressed chunks (same order as input).
        """
        if not chunks:
            return []
        if len(chunks) == 1:
            return [await self.compress(chunks[0])]

        target_ratio = int(self.settings.COLD_TIER_COMPRESSION_RATIO * 100)
        combined_text = "\n---\n".join(
            f"[{i}] {c.text}" for i, c in enumerate(chunks)
        )
        prompt = self.GROUP_COMPRESSION_PROMPT.format(
            count=len(chunks),
            target_ratio=target_ratio,
            combined_text=combined_text,
        )

        # Roughly len(chunks) * per-chunk budget, capped to keep latency bounded
        max_tokens = min(
            self.settings.COMPRESSION_MAX_TOKENS * len(chunks),
            self.settings.LLM_MAX_TOKENS,
        )

        try:
            if self.client._is_anthropic_format():
                response_text = await self.client.complete(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            else:
                response_text = await self.client.complete(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )

            results = self._parse_json_array_response(response_text)
            if len(results) != len(chunks):
                logger.warning(
                    "group_compression_size_mismatch",
                    expected=len(chunks),
                    got=len(results),
                )
                return await self.compress_batch(chunks)

            compressed_chunks: list[CompressedChunk] = []
            for chunk, item in zip(chunks, results):
                summary = item.get("summary", "") if isinstance(item, dict) else ""
                if not summary:
                    # Empty summary on one item - fall back for that chunk only
                    compressed_chunks.append(await self.compress(chunk))
                    continue
                compressed_chunks.append(CompressedChunk(
                    chunk_id=chunk.chunk_id,
                    summary_text=summary,
                    key_entities=item.get("key_entities", []) if isinstance(item, dict) else [],
                    key_facts=item.get("key_facts", []) if isinstance(item, dict) else [],
                    compression_ratio=len(summary) / max(len(chunk.text), 1),
                ))

            logger.info(
                "group_compressed",
                chunk_count=len(chunks),
                avg_ratio=sum(c.compression_ratio for c in compressed_chunks) / len(compressed_chunks),
            )
            return compressed_chunks

        except Exception as e:
            logger.warning("group_compression_failed_falling_back", error=str(e))
            return await self.compress_batch(chunks)

    async def compress_batch(self, chunks: list[Chunk]) -> list[CompressedChunk]:
        """Compress multiple chunks in parallel with rate limiting.

        Args:
            chunks: Chunks to compress.

        Returns:
            List of compressed chunks.
        """
        semaphore = asyncio.Semaphore(self.settings.COMPRESSION_BATCH_SIZE)

        async def compress_one(chunk: Chunk) -> CompressedChunk:
            async with semaphore:
                return await self.compress(chunk)

        return await asyncio.gather(*[
            compress_one(c) for c in chunks
        ])
