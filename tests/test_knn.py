from __future__ import annotations

import pytest

from lc_shift.config import RouterConfig, Strategy
from lc_shift.models import ShiftRequest
from lc_shift.router import RouterShifter
from lc_shift.strategies import KNNStrategy


def _knn_config(three_tier_config: RouterConfig) -> RouterConfig:
    return three_tier_config.model_copy(
        update={
            "strategy": Strategy.KNN,
            "knn_examples": {
                "performance": [
                    "prove the correctness of a consensus algorithm",
                    "analyze and derive time complexity of quicksort",
                    "design a scalable fault tolerant rate limiter",
                ],
                "economy": ["hello there", "tell me a joke", "what is the capital of japan"],
            },
            "knn_k": 3,
        }
    )


class TestKNNStrategy:
    @pytest.mark.asyncio
    async def test_routes_complex_to_performance(self, three_tier_config: RouterConfig) -> None:
        config = _knn_config(three_tier_config)
        strategy = KNNStrategy()
        tier, reason = await strategy.decide(
            ShiftRequest(prompt="derive the complexity and prove correctness of this algorithm"),
            config,
            0.0,
        )
        assert tier == "performance"
        assert "knn vote" in reason

    @pytest.mark.asyncio
    async def test_routes_simple_to_economy(self, three_tier_config: RouterConfig) -> None:
        config = _knn_config(three_tier_config)
        strategy = KNNStrategy()
        tier, _ = await strategy.decide(
            ShiftRequest(prompt="hello there, tell me a joke"), config, 0.0
        )
        assert tier == "economy"

    @pytest.mark.asyncio
    async def test_no_match_returns_default(self, three_tier_config: RouterConfig) -> None:
        config = _knn_config(three_tier_config)
        strategy = KNNStrategy()
        tier, reason = await strategy.decide(ShiftRequest(prompt="zzqq xxyy"), config, 0.0)
        assert tier == config.default_tier
        assert "default tier" in reason

    @pytest.mark.asyncio
    async def test_config_requires_examples(self, three_tier_config: RouterConfig) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="knn_examples must be provided"):
            three_tier_config.model_copy(update={"strategy": Strategy.KNN}).__class__(
                tiers=three_tier_config.tiers,
                default_tier="balanced",
                strategy=Strategy.KNN,
            )


class TestOnlineLearning:
    @pytest.mark.asyncio
    async def test_learn_shifts_routing(self, three_tier_config: RouterConfig) -> None:
        config = _knn_config(three_tier_config)
        router = RouterShifter(config)

        novel = "explain the quantum entanglement protocol in detail"
        before = await router.route(ShiftRequest(prompt=novel))
        # Teach the router that this kind of prompt belongs on performance.
        router.learn("explain the quantum entanglement protocol in detail", "performance")
        after = await router.route(ShiftRequest(prompt=novel))

        assert after.tier_name == "performance"
        # The learned example is now an exact neighbour, so it should win.
        assert after.tier_name != before.tier_name or before.tier_name == "performance"

    @pytest.mark.asyncio
    async def test_learn_rejects_unknown_tier(self, three_tier_config: RouterConfig) -> None:
        from lc_shift.exceptions import RoutingError

        router = RouterShifter(_knn_config(three_tier_config))
        with pytest.raises(RoutingError, match="not in config tiers"):
            router.learn("something", "nonexistent")

    @pytest.mark.asyncio
    async def test_learn_rejects_non_knn_strategy(self, three_tier_config: RouterConfig) -> None:
        from lc_shift.exceptions import RoutingError

        router = RouterShifter(three_tier_config)  # complexity strategy
        with pytest.raises(RoutingError, match="requires the 'knn' strategy"):
            router.learn("something", "performance")


class TestEnsembleStrategy:
    @pytest.mark.asyncio
    async def test_ensemble_votes(self, three_tier_config: RouterConfig) -> None:
        config = three_tier_config.model_copy(
            update={
                "strategy": Strategy.ENSEMBLE,
                "complexity_threshold": 0.3,
                "semantic_routes": {
                    "performance": ["prove and analyze complexity"],
                    "economy": ["hello", "joke"],
                },
                "classifier_weights": {"prove": 2.5, "hello": -2.5},
                "ensemble_weights": {"complexity": 1.0, "semantic": 1.0, "classifier": 0.5},
            }
        )
        router = RouterShifter(config)
        decision = await router.route(
            ShiftRequest(prompt="prove and analyze the complexity of this consensus algorithm")
        )
        assert decision.tier_name == "performance"
        assert "ensemble vote" in decision.reason

    def test_ensemble_requires_weights(self, three_tier_config: RouterConfig) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="ensemble_weights must be provided"):
            RouterConfig(
                tiers=three_tier_config.tiers,
                default_tier="balanced",
                strategy=Strategy.ENSEMBLE,
            )

    def test_ensemble_rejects_unknown_member(self, three_tier_config: RouterConfig) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="not supported"):
            RouterConfig(
                tiers=three_tier_config.tiers,
                default_tier="balanced",
                strategy=Strategy.ENSEMBLE,
                ensemble_weights={"cascade": 1.0},
            )
