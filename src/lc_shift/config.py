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
    SEMANTIC = "semantic"
    CLASSIFIER = "classifier"
    KNN = "knn"
    ENSEMBLE = "ensemble"
    ROLE = "role"


class RouterConfig(BaseModel):
    tiers: dict[str, ModelTier] = Field(min_length=1)
    default_tier: str
    strategy: Strategy = Strategy.COMPLEXITY
    cost_budget_usd: float | None = Field(default=None, ge=0)
    latency_target_ms: float | None = Field(default=None, gt=0)
    complexity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    
    # Semantic Routing config
    semantic_routes: dict[str, list[str]] | None = None
    
    # Classifier Routing config
    classifier_weights: dict[str, float] | None = None
    classifier_intercept: float = 0.0
    classifier_threshold: float = 0.5

    # KNN Routing config — labeled example prompts per tier, k neighbours to vote
    knn_examples: dict[str, list[str]] | None = None
    knn_k: Annotated[int, Field(gt=0)] = 3

    # Ensemble Routing config — strategy name -> vote weight
    ensemble_weights: dict[str, float] | None = None

    # Role Routing config (agent loops) — agent role -> tier name
    role_routes: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_config(self) -> RouterConfig:
        if self.default_tier not in self.tiers:
            raise ValueError(
                f"default_tier '{self.default_tier}' not in tiers: "
                f"{list(self.tiers.keys())}"
            )
            
        if self.strategy == Strategy.SEMANTIC:
            if not self.semantic_routes:
                raise ValueError("semantic_routes must be provided when strategy is 'semantic'")
            for tier_name in self.semantic_routes:
                if tier_name not in self.tiers:
                    raise ValueError(
                        f"semantic route key '{tier_name}' must be a valid tier in tiers: "
                        f"{list(self.tiers.keys())}"
                    )
                    
        if self.strategy == Strategy.CLASSIFIER:
            if not self.classifier_weights:
                raise ValueError("classifier_weights must be provided when strategy is 'classifier'")

        if self.strategy == Strategy.KNN:
            if not self.knn_examples:
                raise ValueError("knn_examples must be provided when strategy is 'knn'")
            for tier_name in self.knn_examples:
                if tier_name not in self.tiers:
                    raise ValueError(
                        f"knn_examples key '{tier_name}' must be a valid tier in tiers: "
                        f"{list(self.tiers.keys())}"
                    )

        if self.strategy == Strategy.ROLE:
            if not self.role_routes:
                raise ValueError("role_routes must be provided when strategy is 'role'")
            for role, tier_name in self.role_routes.items():
                if tier_name not in self.tiers:
                    raise ValueError(
                        f"role_routes['{role}'] -> '{tier_name}' must be a valid tier in tiers: "
                        f"{list(self.tiers.keys())}"
                    )

        if self.strategy == Strategy.ENSEMBLE:
            members = self.ensemble_weights or {}
            if not members:
                raise ValueError("ensemble_weights must be provided when strategy is 'ensemble'")
            allowed = {Strategy.COMPLEXITY.value, Strategy.CLASSIFIER.value, Strategy.SEMANTIC.value}
            for member in members:
                if member not in allowed:
                    raise ValueError(
                        f"ensemble member '{member}' not supported; "
                        f"choose from {sorted(allowed)}"
                    )

        return self

