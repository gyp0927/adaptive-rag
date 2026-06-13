"""Profile API router."""

from fastapi import APIRouter, HTTPException, Query

from hot_and_cold_memory.api.schemas.profile import (
    ProfileFacet,
    ProfileFactResponse,
    ProfileResponse,
)
from hot_and_cold_memory.profile.store import ProfileStore

router = APIRouter(prefix="/profile", tags=["Profile"])

_profile_store: ProfileStore | None = None


def set_profile_store(store: ProfileStore) -> None:
    """Set the global profile store."""
    global _profile_store
    _profile_store = store


@router.get("", response_model=ProfileResponse)
async def get_profile() -> ProfileResponse:
    """Get the current user profile snapshot."""
    if not _profile_store:
        raise HTTPException(status_code=503, detail="Profile store not initialized")

    # TODO: replace with authenticated user_id once auth is implemented
    data = await _profile_store.metadata_store.get_profile("default")
    if not data:
        return ProfileResponse(
            user_id="default",
            identity={},
            preferences=[],
            goals=[],
            constraints=[],
            habits=[],
            attributes={},
            summary=None,
            updated_at=None,
        )

    return ProfileResponse(
        user_id=data["user_id"],
        identity=data.get("identity", {}),
        preferences=[
            ProfileFacet(key=p["key"], value=p["value"], confidence=p.get("confidence"))
            for p in data.get("preferences", [])
        ],
        goals=data.get("goals", []),
        constraints=data.get("constraints", []),
        habits=data.get("habits", []),
        attributes=data.get("attributes", {}),
        summary=data.get("summary"),
        updated_at=data.get("updated_at"),
    )


@router.get("/history", response_model=list[ProfileFactResponse])
async def get_profile_history(
    category: str | None = None,
    key: str | None = None,
    is_current: bool | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[ProfileFactResponse]:
    """Get historical profile facts."""
    if not _profile_store:
        raise HTTPException(status_code=503, detail="Profile store not initialized")

    # TODO: replace with authenticated user_id once auth is implemented
    facts = await _profile_store.list_facts(
        user_id="default",
        category=category,
        key=key,
        is_current=is_current,
        limit=limit,
    )
    return [
        ProfileFactResponse(
            fact_id=f["fact_id"],
            category=f["category"],
            key=f["key"],
            value=f["value"],
            confidence=f["confidence"],
            valid_from=f["valid_from"],
            valid_until=f["valid_until"],
            is_current=f["is_current"],
        )
        for f in facts
    ]
