from __future__ import annotations

import time

from lc_shift.config import RouterConfig, Strategy
from lc_shift.exceptions import BudgetExhaustedError, ConfigurationError, RoutingError
from lc_shift.models import CostSnapshot, RoutingDecision, ShiftRequest
from lc_shift.strategies import STRATEGY_MAP, BaseStrategy


class RouterShifter:
    """Async LLM router that picks a model tier per request."""

    __slots__ = ("_config", "_strategy", "_spent_usd", "_request_counts")

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._strategy: BaseStrategy = self._resolve_strategy(config.strategy)
        self._spent_usd: float = 0.0
        self._request_counts: dict[str, int] = {name: 0 for name in config.tiers}

    @staticmethod
    def _resolve_strategy(strategy: Strategy) -> BaseStrategy:
        cls = STRATEGY_MAP.get(strategy.value)
        if cls is None:
            raise ConfigurationError(f"Unknown strategy: {strategy.value}")
        return cls()

    @property
    def config(self) -> RouterConfig:
        return self._config

    async def route(self, request: ShiftRequest) -> RoutingDecision:
        start_ns = time.perf_counter_ns()

        if request.force_tier is not None:
            if request.force_tier not in self._config.tiers:
                raise RoutingError(
                    f"Forced tier '{request.force_tier}' not in config: "
                    f"{list(self._config.tiers.keys())}"
                )
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            return RoutingDecision(
                tier_name=request.force_tier,
                tier=self._config.tiers[request.force_tier],
                reason="forced by request",
                overhead_ms=elapsed_ms,
            )

        # blow up if we're over budget (cost_aware handles this itself)
        if (
            self._config.cost_budget_usd is not None
            and self._config.strategy != Strategy.COST_AWARE
            and self._spent_usd >= self._config.cost_budget_usd
        ):
            raise BudgetExhaustedError(
                f"Budget of ${self._config.cost_budget_usd:.4f} exhausted "
                f"(spent: ${self._spent_usd:.4f})"
            )

        tier_name, reason = await self._strategy.decide(
            request, self._config, self._spent_usd,
        )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        return RoutingDecision(
            tier_name=tier_name,
            tier=self._config.tiers[tier_name],
            reason=reason,
            overhead_ms=elapsed_ms,
        )

    def record_usage(
        self, tier_name: str, input_tokens: int, output_tokens: int
    ) -> None:
        if tier_name not in self._config.tiers:
            raise RoutingError(f"Unknown tier: {tier_name}")
        tier = self._config.tiers[tier_name]
        cost = (input_tokens / 1000) * tier.cost_per_1k_input + (
            output_tokens / 1000
        ) * tier.cost_per_1k_output
        self._spent_usd += cost
        self._request_counts[tier_name] = self._request_counts.get(tier_name, 0) + 1

    def snapshot(self) -> CostSnapshot:
        return CostSnapshot(
            total_requests=sum(self._request_counts.values()),
            estimated_cost_usd=round(self._spent_usd, 6),
            budget_remaining_usd=(
                round(self._config.cost_budget_usd - self._spent_usd, 6)
                if self._config.cost_budget_usd is not None
                else None
            ),
            requests_by_tier=dict(self._request_counts),
        )

    def reset_tracking(self) -> None:
        self._spent_usd = 0.0
        self._request_counts = {name: 0 for name in self._config.tiers}
