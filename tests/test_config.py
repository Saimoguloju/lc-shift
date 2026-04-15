from __future__ import annotations

import pytest
from pydantic import ValidationError

from lc_shift.config import ModelTier, RouterConfig, Strategy


class TestModelTier:
    def test_valid_tier(self) -> None:
        tier = ModelTier(
            name="Test",
            provider="openai",
            model_id="gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            avg_latency_ms=800,
        )
        assert tier.name == "Test"
        assert tier.max_tokens == 4096

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelTier(
                name="Bad",
                provider="x",
                model_id="x",
                cost_per_1k_input=-1,
                cost_per_1k_output=0,
                avg_latency_ms=100,
            )

    def test_zero_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelTier(
                name="Bad",
                provider="x",
                model_id="x",
                cost_per_1k_input=0,
                cost_per_1k_output=0,
                avg_latency_ms=0,
            )


class TestRouterConfig:
    def test_valid_config(self, three_tier_config: RouterConfig) -> None:
        assert "performance" in three_tier_config.tiers
        assert three_tier_config.default_tier == "balanced"

    def test_default_tier_must_exist(self) -> None:
        tier = ModelTier(
            name="Only",
            provider="x",
            model_id="x",
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            avg_latency_ms=100,
        )
        with pytest.raises(ValidationError, match="default_tier"):
            RouterConfig(
                tiers={"only": tier},
                default_tier="missing",
            )

    def test_empty_tiers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RouterConfig(tiers={}, default_tier="x")

    def test_complexity_threshold_bounds(self) -> None:
        tier = ModelTier(
            name="T",
            provider="x",
            model_id="x",
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            avg_latency_ms=100,
        )
        with pytest.raises(ValidationError):
            RouterConfig(
                tiers={"t": tier},
                default_tier="t",
                complexity_threshold=1.5,
            )

    def test_strategy_enum(self) -> None:
        assert Strategy.COMPLEXITY.value == "complexity"
        assert Strategy.COST_AWARE.value == "cost_aware"
        assert Strategy.CASCADE.value == "cascade"
        assert Strategy.LATENCY.value == "latency"
