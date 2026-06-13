"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Gauge, Histogram

# Query metrics
QUERY_TOTAL = Counter(
    "memory_query_total",
    "Total queries",
    ["tier", "status"],
)

QUERY_DURATION = Histogram(
    "memory_query_duration_seconds",
    "Query latency",
    ["tier"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Memory metrics
MEMORIES_TOTAL = Gauge(
    "memory_items_total",
    "Total memories by tier",
    ["tier"],
)

# Migration metrics
MIGRATION_TOTAL = Counter(
    "memory_migration_total",
    "Total migrations",
    ["direction", "status"],
)

MIGRATION_DURATION = Histogram(
    "memory_migration_duration_seconds",
    "Migration latency",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# LLM metrics
LLM_REQUESTS_TOTAL = Counter(
    "memory_llm_requests_total",
    "Total LLM API calls",
    ["operation"],
)

LLM_REQUEST_DURATION = Histogram(
    "memory_llm_request_duration_seconds",
    "LLM API latency",
    ["operation"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
)

PROFILE_EXTRACTION_TOTAL = Counter(
    "profile_extraction_total", "Total profile extraction attempts", ["status"]
)
PROFILE_QUERY_REWRITES_TOTAL = Counter(
    "profile_query_rewrites_total", "Total profile query rewrites", ["status"]
)
PROFILE_RANKING_BOOSTS_TOTAL = Counter(
    "profile_ranking_boosts_total", "Total profile ranking boosts applied"
)
PROFILE_RECONCILER_RUNS_TOTAL = Counter(
    "profile_reconciler_runs_total", "Total profile reconciler runs", ["status"]
)
