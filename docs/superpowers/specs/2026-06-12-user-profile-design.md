# User Profile Design

**Date:** 2026-06-12  
**Status:** Approved for implementation  
**Topic:** Add user profiling capabilities to the hot-and-cold memory system.

## 1. Goal

Enable the memory system to maintain a structured, time-aware user profile that is automatically extracted from memories on ingestion, exposed via a read-only API, and used to improve retrieval quality through query rewriting and result re-ranking.

## 2. Requirements

### Functional requirements

- Extract structured profile facts from memory content on every memory write and update.
- Support multiple profile dimensions: identity, preferences, goals, constraints, and habits.
- Maintain historical validity of profile facts (e.g., answer "what was my previous job?").
- Provide a read-only API to retrieve the current profile snapshot.
- Use the profile to rewrite retrieval queries for better context understanding.
- Use the profile to boost retrieval results that match the user's stated attributes.
- Support a single user now while keeping the schema extensible for multi-tenant usage.
- Run a periodic background reconciler to resolve conflicts, merge similar facts, and generate a natural-language profile summary.

### Non-functional requirements

- Profile extraction failures must not fail the underlying memory write.
- Query-rewrite failures must fall back to the original query.
- `GET /profile` must be fast (O(1) from a snapshot table).
- The feature must be toggleable via configuration.

## 3. Architecture

Five new modules are added to the existing system:

| Module | Responsibility |
|--------|----------------|
| `ProfileExtractor` | Calls an LLM to extract profile facts from a memory's content. |
| `ProfileBuilder` | Merges new facts, resolves conflicts, and maintains historical validity. |
| `ProfileStore` | Persists profile facts and the current profile snapshot. |
| `ProfileAugmenter` | Rewrites retrieval queries and re-ranks results using the profile. |
| `ProfileReconciler` | Periodically scans facts, resolves contradictions, and regenerates the snapshot summary. |

Integration points:

- **Write path:** `MemoryPipeline` invokes `ProfileExtractor` after a memory is persisted. `ProfileBuilder` and `ProfileStore` update `profile_facts` and `profiles`.
- **Read path:** `GET /api/v1/profile` reads from the `profiles` snapshot table.
- **Retrieval path:** `Retriever` uses `ProfileAugmenter` for query rewriting before search and result re-ranking after search.
- **Background path:** A scheduled task invokes `ProfileReconciler`.

## 4. Data Model

### 4.1 `profile_facts`

Stores every extracted profile assertion with a validity timeline.

```python
class ProfileFactModel(Base):
    __tablename__ = "profile_facts"

    fact_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="default", index=True)
    memory_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("memories.memory_id", ondelete="SET NULL"), nullable=True
    )

    category: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now()
    )
```

### 4.2 `profiles`

A cached snapshot of the current effective profile for fast API reads.

```python
class ProfileModel(Base):
    __tablename__ = "profiles"

    profile_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, default="default")

    identity: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    preferences: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    goals: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    constraints: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    habits: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now()
    )
```

### 4.3 API schemas

```python
class ProfileFacet(BaseModel):
    key: str
    value: Any
    confidence: float | None = None


class ProfileResponse(BaseModel):
    user_id: str
    identity: dict[str, Any]
    preferences: list[ProfileFacet]
    goals: list[str]
    constraints: list[str]
    habits: list[str]
    attributes: dict[str, Any]
    summary: str | None
    updated_at: datetime | None


class ProfileFactResponse(BaseModel):
    fact_id: str
    category: str
    key: str
    value: Any
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    is_current: bool
```

## 5. Data Flow

### 5.1 Write path

```text
Agent writes memory
        │
        ▼
MemoryPipeline persists memory
        │
        ▼
ProfileExtractor calls LLM with memory content
        │
        ▼
ProfileBuilder resolves conflicts
        │   ├─ If (category, key) already has is_current=true, mark it valid_until=now, is_current=false
        │   └─ Insert new fact with is_current=true
        │
        ▼
ProfileStore rebuilds the current snapshot and upserts profiles
```

### 5.2 Retrieval path

```text
Agent sends query
        │
        ▼
ProfileAugmenter loads current profile
        │
        ├─→ Rewrite query using profile context
        │
        ▼
Retriever searches hot/cold tiers
        │
        ▼
ProfileAugmenter computes profile-memory match score
        │
        ├─→ Add profile boost to final ranking
        │
        ▼
Return top_k results
```

### 5.3 Background reconciliation path

```text
Scheduled trigger
        │
        ▼
ProfileReconciler scans recent profile_facts
        │
        ├─ Detect contradictions (e.g., job changed twice in short window)
        ├─ Merge semantically similar preferences
        ├─ Generate natural-language summary
        │
        ▼
Update profiles.summary and bump profiles.version
```

## 6. API Design

### `GET /api/v1/profile`

Returns the current profile snapshot.

**Response:** `ProfileResponse`

### `GET /api/v1/profile/history`

Returns historical profile facts, optionally filtered by `category` and `key`.

**Query parameters:**

- `category` (optional)
- `key` (optional)
- `limit` (optional, default 100)

**Response:** list of `ProfileFactResponse`

### Changes to existing APIs

`POST /api/v1/retrieve` remains unchanged in shape except for one new optional field:

```json
{
  "query": "recommend a web framework",
  "use_profile": false
}
```

`use_profile` defaults to `true`. Profile augmentation is skipped when this flag is `false` or when `ENABLE_PROFILE_AUGMENTATION` is disabled globally.

## 7. Profile Extraction

`ProfileExtractor` sends the memory content to an LLM with a structured-output prompt. The LLM returns zero or more facts:

```json
[
  {
    "category": "identity",
    "key": "role",
    "value": "agent engineer",
    "confidence": 0.95
  }
]
```

Supported categories:

- `identity` — name, role, background
- `preference` — likes, dislikes, style choices
- `goal` — short-term or long-term objectives
- `constraint` — allergies, limitations, must-not-do items
- `habit` — recurring behaviors, active hours, communication patterns

If a memory contains no profile-relevant information, the LLM returns an empty list.

## 8. Conflict Resolution

When a new fact has the same `(user_id, category, key)` as an existing `is_current=true` fact:

1. If the new `value` is identical to the current `value`, do not insert a duplicate; optionally bump the existing fact's confidence.
2. Otherwise, mark the existing fact `valid_until = now()` and `is_current = false`.
3. Insert the new fact with `valid_from = now()` and `is_current = true`.
4. Rebuild the `profiles` snapshot.

This preserves full history and answers "what was my previous X?" queries.

## 9. Retrieval Augmentation

### 9.1 Query rewriting

`ProfileAugmenter` rewrites the user's query by injecting relevant profile context before vector/keyword search. Example:

- Original: `recommend a web framework`
- Rewritten: `recommend a Python web framework suitable for a backend engineer, prefer server-side frameworks`

If rewriting fails, the original query is used.

### 9.2 Result re-ranking

After retrieval, compute a `profile_match_score` for each candidate memory based on overlap between the memory's content/attributes and the current profile. Add this score to the final ranking with a configurable weight:

```text
final_score = vector_score + keyword_score + (profile_match_score * PROFILE_BOOST_WEIGHT)
```

## 10. Error Handling

| Failure | Behavior |
|---------|----------|
| LLM extraction fails | Log warning; memory write succeeds; no facts created. |
| ProfileBuilder sees multiple `is_current=true` facts for same key | Keep newest by `valid_from`; mark others stale. |
| Query rewrite fails | Fall back to original query. |
| Profile snapshot update fails | Log error; retry on next memory write. |
| Profile does not exist | Return empty `ProfileResponse` with HTTP 200. |
| Reconciler fails | Log error; retry on next scheduled run. |

## 11. Configuration

```env
# Master toggle
ENABLE_PROFILE_AUGMENTATION=true

# Sub-toggles
ENABLE_PROFILE_QUERY_REWRITE=true
ENABLE_PROFILE_RANKING_BOOST=true
ENABLE_PROFILE_RECONCILER=true

# Reconciler schedule (cron expression)
PROFILE_RECONCILER_CRON=0 3 * * *

# Ranking boost weight
PROFILE_BOOST_WEIGHT=0.15
```

## 12. Testing

### Unit tests

- `ProfileExtractor` returns expected facts for sample memories and handles empty results.
- `ProfileBuilder` correctly expires old facts and marks new facts current.
- `ProfileStore` rebuilds snapshots from a set of facts.
- `ProfileAugmenter` produces expected rewritten queries and boost scores.
- `ProfileReconciler` detects contradictions and generates summaries.

### Integration tests

- Write a memory → profile facts and snapshot are updated.
- Write a contradictory memory → old fact is expired and API returns the new value.
- Retrieve with profile → rewritten query is used and matching results are boosted.
- LLM extraction fails → memory write still succeeds and system remains healthy.
- Empty profile → `GET /profile` returns 200 with empty fields.

### Edge cases

- Long memory content that may exceed LLM context limits.
- Multiple memories written in one conversation only update relevant profile keys.
- `user_id` isolation for future multi-tenant support.

## 13. Future Work

- Allow manual profile corrections via `PATCH /api/v1/profile`.
- Add confidence decay for old facts similar to memory frequency decay.
- Expose profile-driven memory forgetting (e.g., protect high-importance identity facts).
- Support per-user profile embeddings for semantic retrieval.

## 14. Appendix: File Layout

```text
src/hot_and_cold_memory/
  profile/
    __init__.py
    extractor.py
    builder.py
    store.py
    augmenter.py
    reconciler.py
    schemas.py
  api/routers/
    profile.py
  storage/metadata_store/models.py  # add ProfileFactModel, ProfileModel
```
