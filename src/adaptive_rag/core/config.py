"""Central configuration with Pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from enum import Enum
from typing import Literal


class Tier(str, Enum):
    """Storage tier types."""

    HOT = "hot"
    COLD = "cold"


class VectorDBBackend(str, Enum):
    """Supported vector database backends."""

    QDRANT = "qdrant"


class EmbeddingProvider(str, Enum):
    """Embedding model providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class RoutingStrategy(str, Enum):
    """Query routing strategies."""

    HOT_ONLY = "hot_only"
    COLD_ONLY = "cold_only"
    HOT_FIRST = "hot_first"
    BOTH = "both"


class ChunkStrategy(str, Enum):
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
    APP_NAME: str = "adaptive-rag"
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
    VECTOR_DB_COLLECTION: str = "adaptive_rag"

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

    # Metadata Database
    METADATA_DB_URL: str = "postgresql+asyncpg://rag:rag_password@localhost:5432/adaptive_rag"

    # Cache
    CACHE_URL: str | None = None
    CACHE_TTL_SECONDS: int = 300

    # Document Store
    DOCUMENT_STORE_TYPE: Literal["local"] = "local"
    DOCUMENT_STORE_PATH: str = "./data/documents"

    # Tier Configuration
    COLD_TIER_COMPRESSION_RATIO: float = 0.2
    HOT_TO_COLD_THRESHOLD: float = 0.3
    COLD_TO_HOT_THRESHOLD: float = 0.7

    # Frequency Tracking
    DECAY_HALF_LIFE_HOURS: float = 24.0
    QUERY_CLUSTERING_THRESHOLD: float = 0.85
    MIN_CLUSTER_SIZE: int = 3

    # Chunking
    CHUNK_STRATEGY: ChunkStrategy = ChunkStrategy.RECURSIVE
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Compression
    COMPRESSION_MODEL: str = "gpt-4o-mini"
    COMPRESSION_BATCH_SIZE: int = 10
    COMPRESSION_MAX_TOKENS: int = 256

    # Migration
    MIGRATION_BATCH_SIZE: int = 100
    MIGRATION_INTERVAL_MINUTES: int = 60
    MIGRATION_MAX_CONCURRENT: int = 5

    # LLM
    # 兼容 OpenAI 格式的任意服务商（OpenAI/DeepSeek/通义千问/Kimi等）
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = Field(default="", repr=False)
    DECOMPRESSION_MODEL: str = "gpt-4o"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT_SECONDS: float = 60.0

    # Upload
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

    # Tier capacity
    HOT_TIER_CAPACITY: int = 10000
    HOT_TIER_EVICT_PERCENT: float = 0.1

    # Monitoring
    METRICS_PORT: int = 9090
    ENABLE_TRACING: bool = True


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
