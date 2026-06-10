from __future__ import annotations

import pytest

from lc_shift.agent import AgentRouter, EscalationResult
from lc_shift.config import ModelTier, RouterConfig, Strategy
from lc_shift.models import ShiftRequest
from lc_shift.router import RouterShifter
from lc_shift.strategies import RoleStrategy


class TestRoleStrategy:
    def _role_config(self, three_tier_config: RouterConfig) -> RouterConfig:
        return three_tier_config.model_copy(
            update={
                "strategy": Strategy.ROLE,
                "role_routes": {
                    "planner": "performance",
                    "tool_select": "economy",
                    "summarize": "economy",
                },
            }
        )

    @pytest.mark.asyncio
    async def test_routes_by_role(self, three_tier_config: RouterConfig) -> None:
        config = self._role_config(three_tier_config)
        strat = RoleStrategy()
        tier, reason = await strat.decide(
            ShiftRequest(prompt="plan the steps", metadata={"role": "planner"}), config, 0.0
        )
        assert tier == "performance"
        assert "planner" in reason

    @pytest.mark.asyncio
    async def test_unmapped_role_falls_back(self, three_tier_config: RouterConfig) -> None:
        config = self._role_config(three_tier_config)
        strat = RoleStrategy()
        tier, _ = await strat.decide(
            ShiftRequest(prompt="x", metadata={"role": "unknown"}), config, 0.0
        )
        assert tier == config.default_tier

    @pytest.mark.asyncio
    async def test_no_role_falls_back(self, three_tier_config: RouterConfig) -> None:
        config = self._role_config(three_tier_config)
        strat = RoleStrategy()
        tier, _ = await strat.decide(ShiftRequest(prompt="x"), config, 0.0)
        assert tier == config.default_tier

    def test_config_requires_role_routes(self, three_tier_config: RouterConfig) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="role_routes must be provided"):
            RouterConfig(
                tiers=three_tier_config.tiers, default_tier="balanced", strategy=Strategy.ROLE
            )

    def test_config_rejects_unknown_tier(self, three_tier_config: RouterConfig) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must be a valid tier"):
            RouterConfig(
                tiers=three_tier_config.tiers,
                default_tier="balanced",
                strategy=Strategy.ROLE,
                role_routes={"planner": "ghost"},
            )


class TestAgentRouterRouting:
    @pytest.mark.asyncio
    async def test_route_step_attaches_role(self, three_tier_config: RouterConfig) -> None:
        config = three_tier_config.model_copy(
            update={"strategy": Strategy.ROLE, "role_routes": {"planner": "performance"}}
        )
        agent = AgentRouter(RouterShifter(config))
        decision = await agent.route_step("plan it", role="planner")
        assert decision.tier_name == "performance"

    def test_escalation_order(self, three_tier_config: RouterConfig) -> None:
        agent = AgentRouter(RouterShifter(three_tier_config))
        assert agent.escalation_order() == ["economy", "balanced", "performance"]
        assert agent.escalation_order(start_tier="balanced") == ["balanced", "performance"]

    def test_escalation_order_bad_start(self, three_tier_config: RouterConfig) -> None:
        agent = AgentRouter(RouterShifter(three_tier_config))
        with pytest.raises(ValueError, match="not in tiers"):
            agent.escalation_order(start_tier="ghost")


class TestEscalation:
    @pytest.mark.asyncio
    async def test_accepts_first_when_valid(self, three_tier_config: RouterConfig) -> None:
        agent = AgentRouter(RouterShifter(three_tier_config))
        calls: list[str] = []

        def call(tier: ModelTier, prompt: str) -> str:
            calls.append(tier.model_id)
            return "good answer"

        result = await agent.run_with_escalation("q", call, validate=lambda o: "good" in o)
        assert isinstance(result, EscalationResult)
        assert result.success is True
        assert result.escalated is False
        assert result.tier_name == "economy"  # cheapest, accepted immediately
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_escalates_until_valid(self, three_tier_config: RouterConfig) -> None:
        agent = AgentRouter(RouterShifter(three_tier_config))

        # Only the strongest tier returns an acceptable answer.
        def call(tier: ModelTier, prompt: str) -> str:
            return "ACCEPT" if tier.name == "Performance" else "reject"

        result = await agent.run_with_escalation("q", call, validate=lambda o: o == "ACCEPT")
        assert result.success is True
        assert result.escalated is True
        assert result.tier_name == "performance"
        assert [a.tier_name for a in result.attempts] == ["economy", "balanced", "performance"]
        assert [a.accepted for a in result.attempts] == [False, False, True]

    @pytest.mark.asyncio
    async def test_returns_last_when_none_valid(self, three_tier_config: RouterConfig) -> None:
        agent = AgentRouter(RouterShifter(three_tier_config))
        result = await agent.run_with_escalation(
            "q", lambda tier, p: "nope", validate=lambda o: False
        )
        assert result.success is False
        assert result.tier_name == "performance"  # exhausted to strongest
        assert len(result.attempts) == 3

    @pytest.mark.asyncio
    async def test_async_call_supported(self, three_tier_config: RouterConfig) -> None:
        agent = AgentRouter(RouterShifter(three_tier_config))

        async def call(tier: ModelTier, prompt: str) -> str:
            return f"answer from {tier.model_id}"

        result = await agent.run_with_escalation("q", call)  # no validator -> accept first
        assert result.success is True
        assert result.tier_name == "economy"

    @pytest.mark.asyncio
    async def test_max_attempts_caps_escalation(self, three_tier_config: RouterConfig) -> None:
        agent = AgentRouter(RouterShifter(three_tier_config))
        result = await agent.run_with_escalation(
            "q", lambda tier, p: "x", validate=lambda o: False, max_attempts=2
        )
        assert len(result.attempts) == 2
        assert [a.tier_name for a in result.attempts] == ["economy", "balanced"]
