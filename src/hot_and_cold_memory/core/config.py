"""Central configuration with Pydantic-settings."""

from enum import StrEnum
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Tier(StrEnum):
    """Storage tier types."""

    HOT = "hot"
    COLD = "cold"


class VectorDBBackend(StrEnum):
    """Supported vector database backends."""

    QDRANT = "qdrant"


class EmbeddingProvider(StrEnum):
    """Embedding model providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class RoutingStrategy(StrEnum):
    """Query routing strategies."""

    HOT_ONLY = "hot_only"
    COLD_ONLY = "cold_only"
    HOT_FIRST = "hot_first"
    BOTH = "both"


class ChunkStrategy(StrEnum):
    """Text chunking strategies."""

    RECURSIVE = "recursive"
    LLM = "llm"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "hot-and-cold-memory"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1

    # Vector Database
    VECTOR_DB_BACKEND: VectorDBBackend = VectorDBBackend.QDRANT
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 6333
    VECTOR_DB_COLLECTION: str = "hot_and_cold_memory"

    # Embedding
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.OPENAI
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    # Local embedding model (sentence-transformers)
    # Options: "sentence-transformers/all-MiniLM-L6-v2" (384d)
    #          "sentence-transformers/all-mpnet-base-v2" (768d)
    #          "BAAI/bge-large-zh-v1.5" (1024d, Chinese)
    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_EMBEDDING_DEVICE: str = "cpu"  # "cpu" or "cuda"

    # Metadata Database (no default credentials — must be provided via env)
    METADATA_DB_URL: str = Field(
        ...,
        repr=False,
    )

    # Cache
    CACHE_URL: str | None = None
    CACHE_TTL_SECONDS: int = 300

    # Document Store
    MEMORY_STORE_TYPE: Literal["local"] = "local"
    DOCUMENT_STORE_PATH: str = "./data/memories"

    # Tier Configuration
    COLD_TIER_COMPRESSION_RATIO: float = 0.2
    HOT_TO_COLD_THRESHOLD: float = 0.25
    COLD_TO_HOT_THRESHOLD: float = 0.7
    HOT_ACCESS_COUNT_THRESHOLD: int = 50

    # Frequency Tracking
    DECAY_HALF_LIFE_HOURS: float = 72.0
    QUERY_CLUSTERING_THRESHOLD: float = 0.85
    MIN_CLUSTER_SIZE: int = 3

    # Compression (for long-term memory summarization)
    COMPRESSION_MODEL: str = "gpt-4o-mini"
    COMPRESSION_BATCH_SIZE: int = 10
    COMPRESSION_MAX_TOKENS: int = 256

    # Migration
    MIGRATION_BATCH_SIZE: int = 100
    MIGRATION_INTERVAL_MINUTES: int = 60
    MIGRATION_MAX_CONCURRENT: int = 5

    # Admin / Security
    ADMIN_API_KEY: str | None = None  # If set, required for /admin/* endpoints

    # CORS
    CORS_ALLOW_ORIGINS: str = "*"  # Comma-separated list; "*" for dev only

    # LLM
    # 兼容 OpenAI 格式的任意服务商（OpenAI/DeepSeek/通义千问/Kimi等）
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = Field(..., repr=False)  # Required: no default to avoid runtime failures
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT_SECONDS: float = 60.0

    # Tier capacity
    HOT_TIER_CAPACITY: int = 10000
    HOT_TIER_EVICT_PERCENT: float = 0.1

    # Monitoring
    METRICS_PORT: int = 9090
    ENABLE_TRACING: bool = True

    # Auto-importance scoring
    ENABLE_AUTO_IMPORTANCE: bool = True
    AUTO_IMPORTANCE_USE_LLM: bool = False
    AUTO_IMPORTANCE_LLM_THRESHOLD: float = 0.25

    # Forgetting
    ENABLE_FORGETTING: bool = True
    FORGET_MIN_IMPORTANCE: float = 0.2
    FORGET_MIN_DAYS_SINCE_ACCESS: int = 30
    FORGET_BATCH_SIZE: int = 100

    # Hybrid Search
    ENABLE_HYBRID_SEARCH: bool = True
    HYBRID_RRF_K: int = 60

    # Consolidation (deduplication + merging)
    ENABLE_CONSOLIDATION: bool = True
    CONSOLIDATION_SIMILARITY_THRESHOLD: float = 0.92
    CONSOLIDATION_BATCH_SIZE: int = 50
    CONSOLIDATION_MAX_CANDIDATES: int = 2000
    CONSOLIDATION_MAX_PAIRS_PER_RUN: int = 10
    CONSOLIDATION_MIN_CONTENT_LENGTH: int = 20

    # Association graph
    ENABLE_ASSOCIATIONS: bool = True

    # User Profile
    ENABLE_PROFILE_AUGMENTATION: bool = True
    ENABLE_PROFILE_QUERY_REWRITE: bool = True
    ENABLE_PROFILE_RANKING_BOOST: bool = True
    ENABLE_PROFILE_RECONCILER: bool = True
    PROFILE_RECONCILER_CRON: str = "0 3 * * *"
    PROFILE_BOOST_WEIGHT: float = 0.15
    PROFILE_EXTRACTION_MODEL: str = "gpt-4o-mini"
    PROFILE_MAX_FACTS_PER_MEMORY: int = 10


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
