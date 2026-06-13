# src/hot_and_cold_memory/profile/schemas.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ProfileFact:
    """A single extracted profile assertion."""

    category: str
    key: str
    value: Any
    confidence: float = 0.5


@dataclass
class Profile:
    """Current effective profile snapshot."""

    user_id: str
    identity: dict[str, Any] = field(default_factory=dict)
    preferences: list[dict[str, Any]] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    habits: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    version: int = 1
    updated_at: datetime | None = None

    def is_empty(self) -> bool:
        return not any([
            self.identity,
            self.preferences,
            self.goals,
            self.constraints,
            self.habits,
            self.attributes,
        ])
