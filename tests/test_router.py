from __future__ import annotations

import pytest

from lc_shift.config import ModelTier, RouterConfig, Strategy
from lc_shift.exceptions import BudgetExhaustedError, RoutingError
from lc_shift.models import ShiftRequest
from lc_shift.router import RouterShifter


class TestRouting:
    @pytest.mark.asyncio
    async def test_simple_route(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(three_tier_config)
        decision = await router.route(ShiftRequest(prompt="Hello!"))
        assert decision.tier_name in three_tier_config.tiers
        assert decision.overhead_ms < 50

    @pytest.mark.asyncio
    async def test_overhead_under_50ms(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(three_tier_config)
        for _ in range(100):
            decision = await router.route(ShiftRequest(prompt="quick test"))
            assert decision.overhead_ms < 50

    @pytest.mark.asyncio
    async def test_force_tier(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(three_tier_config)
        decision = await router.route(
            ShiftRequest(prompt="test", force_tier="performance")
        )
        assert decision.tier_name == "performance"
        assert decision.reason == "forced by request"

    @pytest.mark.asyncio
    async def test_force_invalid_tier_raises(
        self, three_tier_config: RouterConfig
    ) -> None:
        router = RouterShifter(three_tier_config)
        with pytest.raises(RoutingError, match="nonexistent"):
            await router.route(
                ShiftRequest(prompt="test", force_tier="nonexistent")
            )

    @pytest.mark.asyncio
    async def test_complex_prompt_routes_to_performance(
        self, three_tier_config: RouterConfig
    ) -> None:
        router = RouterShifter(three_tier_config)
        decision = await router.route(
            ShiftRequest(
                prompt=(
                    "Analyze the following code and explain why this approach "
                    "is better. Compare the trade-off between readability and "
                    "performance.\n```python\ndef solve(): pass\n```\n"
                    "1. First evaluate 2. Then assess 3. Finally justify"
                )
            )
        )
        assert decision.tier_name == "performance"


class TestCostTracking:
    @pytest.mark.asyncio
    async def test_record_and_snapshot(
        self, three_tier_config: RouterConfig
    ) -> None:
        router = RouterShifter(three_tier_config)
        router.record_usage("economy", input_tokens=1000, output_tokens=500)
        snap = router.snapshot()
        assert snap.total_requests == 1
        assert snap.estimated_cost_usd > 0
        assert snap.requests_by_tier["economy"] == 1

    def test_record_unknown_tier_raises(
        self, three_tier_config: RouterConfig
    ) -> None:
        router = RouterShifter(three_tier_config)
        with pytest.raises(RoutingError, match="Unknown tier"):
            router.record_usage("fake", input_tokens=100, output_tokens=50)

    @pytest.mark.asyncio
    async def test_reset_tracking(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(three_tier_config)
        router.record_usage("economy", input_tokens=1000, output_tokens=500)
        router.reset_tracking()
        snap = router.snapshot()
        assert snap.total_requests == 0
        assert snap.estimated_cost_usd == 0.0


class TestBudgetGuard:
    @pytest.mark.asyncio
    async def test_budget_exhausted_raises(self) -> None:
        config = RouterConfig(
            tiers={
                "cheap": ModelTier(
                    name="Cheap",
                    provider="x",
                    model_id="x",
                    cost_per_1k_input=0.001,
                    cost_per_1k_output=0.002,
                    avg_latency_ms=100,
                ),
            },
            default_tier="cheap",
            strategy=Strategy.COMPLEXITY,
            cost_budget_usd=0.01,
        )
        router = RouterShifter(config)
        router.record_usage("cheap", input_tokens=10_000, output_tokens=5_000)

        with pytest.raises(BudgetExhaustedError):
            await router.route(ShiftRequest(prompt="over budget"))

    @pytest.mark.asyncio
    async def test_cost_aware_degrades_gracefully(
        self, budget_config: RouterConfig
    ) -> None:
        router = RouterShifter(budget_config)
        router.record_usage("performance", input_tokens=50_000, output_tokens=10_000)
        # should route to a cheaper tier, not blow up
        decision = await router.route(ShiftRequest(prompt="still going"))
        assert decision.tier_name in budget_config.tiers
