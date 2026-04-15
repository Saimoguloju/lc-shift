from __future__ import annotations

from pydantic import BaseModel, Field

from lc_shift.config import ModelTier


class ShiftRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, str] = Field(default_factory=dict)
    force_tier: str | None = None


class RoutingDecision(BaseModel):
    tier_name: str
    tier: ModelTier
    reason: str
    overhead_ms: float = Field(ge=0)


class CostSnapshot(BaseModel):
    total_requests: int = 0
    estimated_cost_usd: float = 0.0
    budget_remaining_usd: float | None = None
    requests_by_tier: dict[str, int] = Field(default_factory=dict)
