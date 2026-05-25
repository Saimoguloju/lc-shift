"""RouterShifter — core routing orchestrator."""

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
from lc_shift.strategies import STRATEGY_MAP, BaseStrategy


class RouterShifter:
    """Provider-agnostic LLM tier router.

    Makes sub-millisecond routing decisions based on prompt complexity,
    budget state, latency targets, or cascade priority — without calling
    any LLM API itself.

    Features
    --------
    - ``route()`` — single routing decision
    - ``route_with_fallback()`` — ordered fallback chain, degraded tiers excluded
    - ``route_batch()`` — concurrent routing for a list of requests
    - ``mark_tier_failed()`` — signal a provider error; tier skipped for cooldown
    - ``recover_tier()`` — manually clear a tier's degraded state
    - ``record_usage()`` — update cost tracking after your LLM call
    - ``snapshot()`` — rich metrics snapshot with per-tier stats and cache hit rate
    - Observability hooks (``HookRegistry``) for logging / alerting / tracing
    - In-memory decision cache (``RoutingCache``) for identical prompts
    - Async context manager support

    Basic usage::

        config = RouterConfig(tiers=PRESETS["anthropic-3tier"], default_tier="balanced")
        async with RouterShifter(config) as router:
            decision = await router.route(ShiftRequest(prompt="Hello"))
            result = await call_your_llm(decision.tier)
            router.record_usage(decision.tier_name, input_tokens=50, output_tokens=200)

    With hooks and cache::

        hooks = HookRegistry()

        @hooks.on_route
        def log_decision(request, decision):
            print(f"[{decision.tier_name}] {decision.overhead_ms:.2f}ms — {decision.reason}")

        @hooks.on_fallback
        async def alert(failed_tier, next_tier, exc):
            await notify_team(f"{failed_tier} degraded → {next_tier}")

        cache = RoutingCache(ttl_seconds=120)
        router = RouterShifter(config, hooks=hooks, cache=cache)
    """

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
        cls = STRATEGY_MAP.get(config.strategy.value)
        if cls is None:
            raise ConfigurationError(f"Unknown strategy: {config.strategy.value}")

        self._config = config
        self._strategy: BaseStrategy = cls()
        self._spent_usd: float = 0.0
        self._request_counts: dict[str, int] = {n: 0 for n in config.tiers}
        self._tier_metrics: dict[str, TierMetrics] = {
            n: TierMetrics(tier_name=n) for n in config.tiers
        }
        self._total_cache_hits: int = 0
        self._health: TierHealth = health or TierHealth()
        self._cache: RoutingCache | None = cache
        self._hooks: HookRegistry | None = hooks

    # -------------------------------------------------------------------------
    # Single routing decision
    # -------------------------------------------------------------------------

    async def route(self, request: ShiftRequest) -> RoutingDecision:
        """Return the best tier for *request*.

        Raises ``BudgetExhaustedError`` when the budget is fully consumed.
        Raises ``RoutingError`` for invalid ``force_tier`` values.
        """
        self._check_budget()

        if request.force_tier is not None:
            return self._make_forced(request)

        # Cache lookup — identical prompts avoid re-scoring entirely
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

    # -------------------------------------------------------------------------
    # Fallback chain
    # -------------------------------------------------------------------------

    async def route_with_fallback(self, request: ShiftRequest) -> FallbackChain:
        """Return an ordered fallback chain of healthy tiers.

        The first entry is the primary strategy decision. Degraded tiers are
        skipped automatically and listed in ``FallbackChain.skipped_tiers``.
        Remaining tiers are appended sorted cheapest-last so the chain degrades
        gracefully in quality while spending less.

        Example::

            chain = await router.route_with_fallback(request)
            for decision in chain:
                try:
                    result = await call_llm(decision.tier)
                    router.record_usage(decision.tier_name, ...)
                    break
                except ProviderError as exc:
                    router.mark_tier_failed(decision.tier_name)
                    if router.hooks:
                        await router.hooks.fire_fallback(
                            decision.tier_name, "next", exc
                        )
            else:
                raise RuntimeError("All tiers exhausted")
        """
        self._check_budget()
        skipped: list[str] = []
        decisions: list[RoutingDecision] = []

        # Primary decision, skipping any degraded tier the strategy returns
        for _ in range(len(self._config.tiers)):
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
                break
            if tier_name not in skipped:
                skipped.append(tier_name)
        # Append remaining healthy tiers as further fallbacks, cheapest last
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
            else:
                if tier_name not in skipped:
                    skipped.append(tier_name)

        if not decisions:
            raise RoutingError(
                f"All tiers are degraded — no healthy fallback. "
                f"Skipped: {skipped}"
            )

        return FallbackChain(tiers=decisions, skipped_tiers=skipped)

    # -------------------------------------------------------------------------
    # Batch routing
    # -------------------------------------------------------------------------

    async def route_batch(
        self, requests: list[ShiftRequest]
    ) -> list[RoutingDecision]:
        """Route multiple requests concurrently, preserving order.

        Example::

            decisions = await router.route_batch([
                ShiftRequest(prompt="Simple question"),
                ShiftRequest(prompt="Analyze this complex architecture..."),
                ShiftRequest(prompt="Translate to French"),
            ])
        """
        return list(await asyncio.gather(*[self.route(r) for r in requests]))

    # -------------------------------------------------------------------------
    # Tier health management
    # -------------------------------------------------------------------------

    def mark_tier_failed(self, tier_name: str) -> None:
        """Signal a provider error on *tier_name*.

        The tier is excluded from routing for ``TierHealth.cooldown_seconds``
        (default 60 s) and then automatically recovered.
        """
        self._health.mark_failed(tier_name)

    def recover_tier(self, tier_name: str) -> None:
        """Manually clear a tier's cooldown before it expires."""
        self._health.recover(tier_name)

    # -------------------------------------------------------------------------
    # Usage tracking
    # -------------------------------------------------------------------------

    def record_usage(
        self,
        tier_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Update cost accounting after your LLM API call completes."""
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
            asyncio.get_event_loop().create_task(
                self._hooks.fire_usage(tier_name, input_tokens, output_tokens)
            )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def snapshot(self) -> CostSnapshot:
        """Return a rich metrics snapshot.

        Includes per-tier request counts, token usage, cost breakdown,
        cache hit rate, and list of currently degraded tiers.
        """
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
        """Clear all cost, request, and cache state."""
        self._spent_usd = 0.0
        self._request_counts = {n: 0 for n in self._config.tiers}
        self._tier_metrics = {
            n: TierMetrics(tier_name=n) for n in self._config.tiers
        }
        self._total_cache_hits = 0
        if self._cache is not None:
            self._cache.clear()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def config(self) -> RouterConfig:
        return self._config

    @property
    def health(self) -> TierHealth:
        return self._health

    @property
    def hooks(self) -> HookRegistry | None:
        return self._hooks

    # -------------------------------------------------------------------------
    # Async context manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> RouterShifter:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass  # reserved for future HTTP session / connection cleanup

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _check_budget(self) -> None:
        budget = self._config.cost_budget_usd
        if budget is not None and self._config.strategy != Strategy.COST_AWARE:
            if self._spent_usd >= budget:
                raise BudgetExhaustedError(
                    f"Budget of ${budget:.4f} exhausted "
                    f"(spent ${self._spent_usd:.4f})"
                )

    def _make_forced(self, request: ShiftRequest) -> RoutingDecision:
        tier_name = request.force_tier  # type: ignore[assignment]
        if tier_name not in self._config.tiers:
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
