from __future__ import annotations

import pytest

from lc_shift.config import RouterConfig, Strategy
from lc_shift.models import ShiftRequest
from lc_shift.strategies import (
    CascadeStrategy,
    ComplexityStrategy,
    CostAwareStrategy,
    LatencyStrategy,
    compute_complexity,
    estimate_token_count,
)


class TestComplexityScoring:
    def test_short_simple_prompt(self) -> None:
        assert compute_complexity("Hello, how are you?") < 0.3

    def test_long_prompt_increases_score(self) -> None:
        assert compute_complexity("word " * 400) >= 0.2

    def test_code_block_increases_score(self) -> None:
        prompt = "Fix this:\n```python\ndef foo(): pass\n```"
        assert compute_complexity(prompt) >= 0.25

    def test_reasoning_keywords_increase_score(self) -> None:
        prompt = "Analyze and compare the trade-off between these approaches. Explain why one is better."
        assert compute_complexity(prompt) >= 0.15

    def test_multi_part_increases_score(self) -> None:
        prompt = "1. First do X\n2. Then do Y\n3. Finally do Z"
        assert compute_complexity(prompt) > 0

    def test_score_capped_at_one(self) -> None:
        prompt = (
            "Analyze and compare and evaluate and explain the trade-off. "
            "1. First 2. Second 3. Third 4. Fourth 5. Fifth "
            "```python\nprint('hello')\n``` " * 5
            + "word " * 500
        )
        assert compute_complexity(prompt) <= 1.0


class TestTokenEstimate:
    def test_single_word(self) -> None:
        assert estimate_token_count("hello") >= 1

    def test_scales_with_length(self) -> None:
        short = estimate_token_count("hello world")
        long = estimate_token_count("hello world " * 100)
        assert long > short


class TestComplexityStrategy:
    @pytest.mark.asyncio
    async def test_simple_prompt_routes_to_cheapest(
        self, three_tier_config: RouterConfig
    ) -> None:
        strategy = ComplexityStrategy()
        req = ShiftRequest(prompt="Hi there")
        tier_name, _ = await strategy.decide(req, three_tier_config, 0.0)
        assert tier_name == "economy"

    @pytest.mark.asyncio
    async def test_complex_prompt_routes_to_performance(
        self, three_tier_config: RouterConfig
    ) -> None:
        strategy = ComplexityStrategy()
        req = ShiftRequest(
            prompt=(
                "Analyze and compare the following code. Explain why one approach "
                "is better. ```python\ndef solve(): pass\n```\n"
                "1. First evaluate correctness\n2. Then assess performance"
            )
        )
        tier_name, _ = await strategy.decide(req, three_tier_config, 0.0)
        assert tier_name == "performance"


class TestCostAwareStrategy:
    @pytest.mark.asyncio
    async def test_healthy_budget_routes_expensive(
        self, budget_config: RouterConfig
    ) -> None:
        strategy = CostAwareStrategy()
        req = ShiftRequest(prompt="test prompt")
        tier_name, reason = await strategy.decide(req, budget_config, 0.0)
        assert tier_name == "performance"
        assert "healthy" in reason

    @pytest.mark.asyncio
    async def test_exhausted_budget_routes_cheapest(
        self, budget_config: RouterConfig
    ) -> None:
        strategy = CostAwareStrategy()
        req = ShiftRequest(prompt="test prompt")
        tier_name, reason = await strategy.decide(req, budget_config, 1.50)
        assert tier_name == "economy"
        assert "exhausted" in reason

    @pytest.mark.asyncio
    async def test_moderate_budget(self, budget_config: RouterConfig) -> None:
        strategy = CostAwareStrategy()
        req = ShiftRequest(prompt="test prompt")
        _, reason = await strategy.decide(req, budget_config, 0.60)
        assert "moderate" in reason

    @pytest.mark.asyncio
    async def test_no_budget_returns_default(
        self, three_tier_config: RouterConfig
    ) -> None:
        strategy = CostAwareStrategy()
        req = ShiftRequest(prompt="test prompt")
        tier_name, _ = await strategy.decide(req, three_tier_config, 0.0)
        assert tier_name == "balanced"


class TestCascadeStrategy:
    @pytest.mark.asyncio
    async def test_always_returns_cheapest(
        self, three_tier_config: RouterConfig
    ) -> None:
        strategy = CascadeStrategy()
        req = ShiftRequest(prompt="test prompt")
        tier_name, reason = await strategy.decide(req, three_tier_config, 0.0)
        assert tier_name == "economy"
        assert "cascade" in reason


class TestLatencyStrategy:
    @pytest.mark.asyncio
    async def test_picks_best_under_target(
        self, latency_config: RouterConfig
    ) -> None:
        strategy = LatencyStrategy()
        req = ShiftRequest(prompt="test prompt")
        # target is 1500ms; balanced (1200) and economy (400) both qualify,
        # should pick balanced since it's more capable
        tier_name, _ = await strategy.decide(req, latency_config, 0.0)
        assert tier_name == "balanced"

    @pytest.mark.asyncio
    async def test_falls_back_to_fastest_when_none_qualify(
        self, three_tier_config: RouterConfig
    ) -> None:
        config = three_tier_config.model_copy(
            update={"strategy": Strategy.LATENCY, "latency_target_ms": 100}
        )
        strategy = LatencyStrategy()
        req = ShiftRequest(prompt="test prompt")
        tier_name, reason = await strategy.decide(req, config, 0.0)
        assert tier_name == "economy"
        assert "no tier meets" in reason

    @pytest.mark.asyncio
    async def test_no_target_returns_default(
        self, three_tier_config: RouterConfig
    ) -> None:
        strategy = LatencyStrategy()
        req = ShiftRequest(prompt="test prompt")
        tier_name, _ = await strategy.decide(req, three_tier_config, 0.0)
        assert tier_name == "balanced"


class TestLocalTFIDF:
    def test_fit_and_transform(self) -> None:
        from lc_shift.strategies import LocalTFIDF
        tfidf = LocalTFIDF()
        docs = [
            "write some python code for sorting list",
            "explain the mathematical proof of theory",
            "say a simple hello to user"
        ]
        tfidf.fit(docs)
        assert len(tfidf.vocab) > 0
        
        v1 = tfidf.transform("write python code")
        v2 = tfidf.transform("explain mathematical proof")
        v3 = tfidf.transform("write python code")
        
        sim_same = LocalTFIDF.cosine_similarity(v1, v3)
        sim_diff = LocalTFIDF.cosine_similarity(v1, v2)
        
        assert sim_same == pytest.approx(1.0)
        assert sim_diff < 0.2


class TestSemanticStrategy:
    @pytest.mark.asyncio
    async def test_semantic_routing_selection(
        self, three_tier_config: RouterConfig
    ) -> None:
        from lc_shift.strategies import SemanticStrategy
        config = three_tier_config.model_copy(
            update={
                "strategy": Strategy.SEMANTIC,
                "semantic_routes": {
                    "performance": ["write some complex rust code for memory management", "compile compiler optimization"],
                    "economy": ["hi", "hello there", "what's up"]
                }
            }
        )
        strategy = SemanticStrategy()
        
        req_code = ShiftRequest(prompt="implement compiler rust code optimizations")
        tier_code, reason_code = await strategy.decide(req_code, config, 0.0)
        assert tier_code == "performance"
        assert "semantic match" in reason_code
        
        req_greet = ShiftRequest(prompt="hello there friend")
        tier_greet, reason_greet = await strategy.decide(req_greet, config, 0.0)
        assert tier_greet == "economy"
        
        req_unrelated = ShiftRequest(prompt="xyzabc123")
        tier_unrelated, reason_unrelated = await strategy.decide(req_unrelated, config, 0.0)
        assert tier_unrelated == config.default_tier


class TestClassifierStrategy:
    @pytest.mark.asyncio
    async def test_classifier_routing_selection(
        self, three_tier_config: RouterConfig
    ) -> None:
        from lc_shift.strategies import ClassifierStrategy
        config = three_tier_config.model_copy(
            update={
                "strategy": Strategy.CLASSIFIER,
                "classifier_weights": {
                    "code": 2.5,
                    "optimize": 1.5,
                    "hi": -2.0,
                    "simple": -1.5
                },
                "classifier_intercept": -0.5,
                "classifier_threshold": 0.6
            }
        )
        strategy = ClassifierStrategy()
        
        req_perf = ShiftRequest(prompt="code optimize")
        tier_perf, reason_perf = await strategy.decide(req_perf, config, 0.0)
        assert tier_perf == "performance"
        assert "classifier probability" in reason_perf
        
        req_econ = ShiftRequest(prompt="hi simple")
        tier_econ, reason_econ = await strategy.decide(req_econ, config, 0.0)
        assert tier_econ == "economy"
