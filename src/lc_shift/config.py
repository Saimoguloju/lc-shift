from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class ModelTier(BaseModel):
    name: str
    provider: str
    model_id: str
    cost_per_1k_input: Annotated[float, Field(ge=0)]
    cost_per_1k_output: Annotated[float, Field(ge=0)]
    avg_latency_ms: Annotated[float, Field(gt=0)]
    max_tokens: Annotated[int, Field(gt=0)] = 4096


class Strategy(str, Enum):
    COMPLEXITY = "complexity"
    COST_AWARE = "cost_aware"
    CASCADE = "cascade"
    LATENCY = "latency"


class RouterConfig(BaseModel):
    tiers: dict[str, ModelTier] = Field(min_length=1)
    default_tier: str
    strategy: Strategy = Strategy.COMPLEXITY
    cost_budget_usd: float | None = Field(default=None, ge=0)
    latency_target_ms: float | None = Field(default=None, gt=0)
    complexity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    @model_validator(mode="after")
    def _validate_default_tier(self) -> RouterConfig:
        if self.default_tier not in self.tiers:
            raise ValueError(
                f"default_tier '{self.default_tier}' not in tiers: "
                f"{list(self.tiers.keys())}"
            )
        return self
