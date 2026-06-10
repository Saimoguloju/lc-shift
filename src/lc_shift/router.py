from __future__ import annotations

import asyncio
import time

from lc_shift.cache import RoutingCache
from lc_shift.config import RouterConfig, Strategy
from lc_shift.exceptions import BudgetExhaustedError, ConfigurationError, RoutingError
from lc_shift.health import TierHealth
from lc_shift.hooks import HookRegistry
from lc_shift.models import (
    CostSnapshot,
    FallbackChain,
    RoutingDecision,
    ShiftRequest,
    TierMetrics,
)
from lc_shift.strategies import STRATEGY_MAP, BaseStrategy, KNNStrategy


class RouterShifter:
    """Orchestrates LLM prompt routing across configured model tiers."""

    __slots__ = (
        "_config",
        "_strategy",
        "_spent_usd",
        "_request_counts",
        "_tier_metrics",
        "_total_cache_hits",
        "_health",
        "_cache",
        "_hooks",
    )

    def __init__(
        self,
        config: RouterConfig,
        *,
        health: TierHealth | None = None,
        cache: RoutingCache | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        strategy_class = STRATEGY_MAP.get(config.strategy.value)
        if strategy_class is None:
            raise ConfigurationError(f"Unknown strategy: {config.strategy.value}")

        self._config = config
        self._strategy: BaseStrategy = strategy_class()
        self._spent_usd: float = 0.0
        self._request_counts: dict[str, int] = {n: 0 for n in config.tiers}
        self._tier_metrics: dict[str, TierMetrics] = {
            n: TierMetrics(tier_name=n) for n in config.tiers
        }
        self._total_cache_hits: int = 0
        self._health: TierHealth = health or TierHealth()
        self._cache: RoutingCache | None = cache
        self._hooks: HookRegistry | None = hooks

    async def route(self, request: ShiftRequest) -> RoutingDecision:
        """Determines the best tier for the given request."""
        self._check_budget()

        if request.force_tier is not None:
            return self._make_forced(request)

        if self._cache is not None:
            cached = self._cache.get(request.prompt, self._config.strategy.value)
            if cached is not None:
                self._total_cache_hits += 1
                hit = cached.model_copy(update={"cache_hit": True, "overhead_ms": 0.0})
                if self._hooks:
                    await self._hooks.fire_route(request, hit)
                return hit

        t0 = time.perf_counter_ns()
        tier_name, reason = await self._strategy.decide(
            request, self._config, self._spent_usd
        )
        overhead_ms = (time.perf_counter_ns() - t0) / 1_000_000

        decision = RoutingDecision(
            tier_name=tier_name,
            tier=self._config.tiers[tier_name],
            reason=reason,
            overhead_ms=overhead_ms,
        )

        if self._cache is not None:
            self._cache.set(request.prompt, self._config.strategy.value, decision)

        if self._hooks:
            await self._hooks.fire_route(request, decision)

        return decision

    async def route_with_fallback(self, request: ShiftRequest) -> FallbackChain:
        """Returns an ordered fallback chain of healthy tiers."""
        self._check_budget()
        skipped: list[str] = []
        decisions: list[RoutingDecision] = []

        t0 = time.perf_counter_ns()
        tier_name, reason = await self._strategy.decide(
            request, self._config, self._spent_usd
        )
        overhead_ms = (time.perf_counter_ns() - t0) / 1_000_000

        if self._health.is_healthy(tier_name):
            decisions.append(
                RoutingDecision(
                    tier_name=tier_name,
                    tier=self._config.tiers[tier_name],
                    reason=reason,
                    overhead_ms=overhead_ms,
                )
            )
        else:
            skipped.append(tier_name)

        seen = {d.tier_name for d in decisions} | set(skipped)
        remaining = sorted(
            [(n, t) for n, t in self._config.tiers.items() if n not in seen],
            key=lambda x: x[1].cost_per_1k_input,
            reverse=True,
        )
        for tier_name, tier in remaining:
            if self._health.is_healthy(tier_name):
                decisions.append(
                    RoutingDecision(
                        tier_name=tier_name,
                        tier=tier,
                        reason="fallback candidate",
                        overhead_ms=0.0,
                    )
                )
            elif tier_name not in skipped:
                skipped.append(tier_name)

        if not decisions:
            raise RoutingError(
                f"All tiers are degraded — no healthy fallback. Skipped: {skipped}"
            )

        return FallbackChain(tiers=decisions, skipped_tiers=skipped)

    async def route_batch(
        self, requests: list[ShiftRequest]
    ) -> list[RoutingDecision]:
        """Routes multiple requests concurrently, preserving order."""
        return list(await asyncio.gather(*[self.route(r) for r in requests]))

    def learn(self, prompt: str, tier_name: str) -> None:
        """Teach the router from feedback (online learning).

        Adds a labelled example so future, semantically-similar prompts route to
        ``tier_name``. Only meaningful for the KNN strategy; raises for others so
        misuse surfaces early. The next decision rebuilds the index lazily, and
        cached decisions for the strategy are invalidated.
        """
        if tier_name not in self._config.tiers:
            raise RoutingError(
                f"learn() tier '{tier_name}' not in config tiers: "
                f"{list(self._config.tiers.keys())}"
            )
        if not isinstance(self._strategy, KNNStrategy):
            raise RoutingError(
                f"learn() requires the 'knn' strategy, but router uses "
                f"'{self._config.strategy.value}'"
            )
        self._strategy.learn(prompt, tier_name)
        if self._cache is not None:
            self._cache.clear()

    def mark_tier_failed(self, tier_name: str) -> None:
        self._health.mark_failed(tier_name)

    def recover_tier(self, tier_name: str) -> None:
        self._health.recover(tier_name)

    def record_usage(
        self,
        tier_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Logs costs and token usage statistics for a specific tier."""
        if tier_name not in self._config.tiers:
            raise RoutingError(f"Unknown tier '{tier_name}'")
        tier = self._config.tiers[tier_name]
        cost = (input_tokens / 1000) * tier.cost_per_1k_input + (
            output_tokens / 1000
        ) * tier.cost_per_1k_output
        self._spent_usd += cost
        self._request_counts[tier_name] = self._request_counts.get(tier_name, 0) + 1
        m = self._tier_metrics[tier_name]
        m.requests += 1
        m.estimated_cost_usd += cost
        m.input_tokens += input_tokens
        m.output_tokens += output_tokens

        if self._hooks:
            self._hooks.dispatch_usage(tier_name, input_tokens, output_tokens)

    def snapshot(self) -> CostSnapshot:
        """Returns a metrics snapshot of the current router state."""
        budget = self._config.cost_budget_usd
        total = sum(self._request_counts.values())
        cache_hit_rate = (self._total_cache_hits / total) if total > 0 else 0.0

        return CostSnapshot(
            total_requests=total,
            estimated_cost_usd=round(self._spent_usd, 6),
            budget_remaining_usd=(
                round(budget - self._spent_usd, 6) if budget is not None else None
            ),
            requests_by_tier=dict(self._request_counts),
            tier_metrics=dict(self._tier_metrics),
            cache_hit_rate=round(cache_hit_rate, 4),
            degraded_tiers=self._health.degraded_tiers,
        )

    def reset_tracking(self) -> None:
        self._spent_usd = 0.0
        self._request_counts = {n: 0 for n in self._config.tiers}
        self._tier_metrics = {
            n: TierMetrics(tier_name=n) for n in self._config.tiers
        }
        self._total_cache_hits = 0
        if self._cache is not None:
            self._cache.clear()

    @property
    def config(self) -> RouterConfig:
        return self._config

    @property
    def spent_usd(self) -> float:
        """Total estimated spend recorded so far, in USD."""
        return self._spent_usd

    def seed_spend(self, amount_usd: float) -> None:
        """Set the cumulative spend directly.

        Useful for restoring router state across stateless invocations (e.g. a
        web UI that recreates the router each rerun). Cost-aware routing and
        budget guards read this value.
        """
        if amount_usd < 0:
            raise RoutingError(f"seed_spend amount must be >= 0, got {amount_usd}")
        self._spent_usd = amount_usd

    @property
    def health(self) -> TierHealth:
        return self._health

    @property
    def hooks(self) -> HookRegistry | None:
        return self._hooks

    async def __aenter__(self) -> RouterShifter:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    def _check_budget(self) -> None:
        budget = self._config.cost_budget_usd
        if budget is not None and self._config.strategy != Strategy.COST_AWARE:
            if self._spent_usd >= budget:
                raise BudgetExhaustedError(
                    f"Budget of ${budget:.4f} exhausted (spent ${self._spent_usd:.4f})"
                )

    def _make_forced(self, request: ShiftRequest) -> RoutingDecision:
        tier_name = request.force_tier
        if not tier_name or tier_name not in self._config.tiers:
            raise RoutingError(
                f"force_tier '{tier_name}' not in config tiers: "
                f"{list(self._config.tiers.keys())}"
            )
        return RoutingDecision(
            tier_name=tier_name,
            tier=self._config.tiers[tier_name],
            reason="force_tier override",
            overhead_ms=0.0,
        )
