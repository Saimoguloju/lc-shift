from __future__ import annotations

import pytest

from lc_shift.config import ModelTier, RouterConfig, Strategy


def _make_tiers() -> dict[str, ModelTier]:
    return {
        "performance": ModelTier(
            name="Performance",
            provider="anthropic",
            model_id="claude-opus-4-6",
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            avg_latency_ms=2500,
        ),
        "balanced": ModelTier(
            name="Balanced",
            provider="anthropic",
            model_id="claude-sonnet-4-6",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            avg_latency_ms=1200,
        ),
        "economy": ModelTier(
            name="Economy",
            provider="anthropic",
            model_id="claude-haiku-4-5",
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
            avg_latency_ms=400,
        ),
    }


@pytest.fixture()
def three_tier_config() -> RouterConfig:
    return RouterConfig(
        tiers=_make_tiers(),
        default_tier="balanced",
        strategy=Strategy.COMPLEXITY,
    )


@pytest.fixture()
def budget_config() -> RouterConfig:
    return RouterConfig(
        tiers=_make_tiers(),
        default_tier="balanced",
        strategy=Strategy.COST_AWARE,
        cost_budget_usd=1.00,
    )


@pytest.fixture()
def latency_config() -> RouterConfig:
    return RouterConfig(
        tiers=_make_tiers(),
        default_tier="balanced",
        strategy=Strategy.LATENCY,
        latency_target_ms=1500,
    )
