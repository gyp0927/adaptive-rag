"""API request/response schemas for user profiles."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProfileFacet(BaseModel):
    """A single preference facet with confidence."""

    model_config = ConfigDict(from_attributes=True)

    key: str
    value: Any
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ProfileResponse(BaseModel):
    """Current user profile snapshot."""

    model_config = ConfigDict(from_attributes=True)

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
    """A historical profile fact."""

    model_config = ConfigDict(from_attributes=True)

    fact_id: str
    category: str
    key: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: datetime
    valid_until: datetime | None
    is_current: bool
