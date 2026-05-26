from __future__ import annotations

from typing import Iterator
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
    cache_hit: bool = False


class FallbackChain(BaseModel):
    """Ordered fallback chain representing fallback routing results."""

    tiers: list[RoutingDecision]
    skipped_tiers: list[str] = Field(default_factory=list)

    @property
    def primary(self) -> RoutingDecision:
        return self.tiers[0]

    def __iter__(self) -> Iterator[RoutingDecision]:  # type: ignore[override]
        return iter(self.tiers)

    def __len__(self) -> int:
        return len(self.tiers)


class TierMetrics(BaseModel):
    tier_name: str
    requests: int = 0
    estimated_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0


class CostSnapshot(BaseModel):
    total_requests: int = 0
    estimated_cost_usd: float = 0.0
    budget_remaining_usd: float | None = None
    requests_by_tier: dict[str, int] = Field(default_factory=dict)
    tier_metrics: dict[str, TierMetrics] = Field(default_factory=dict)
    cache_hit_rate: float = 0.0
    degraded_tiers: list[str] = Field(default_factory=list)
