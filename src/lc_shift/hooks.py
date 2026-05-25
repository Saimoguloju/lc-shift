"""Observability hooks — pluggable callbacks fired on routing events."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Union

from lc_shift.models import RoutingDecision, ShiftRequest

_MaybeAwaitable = Union[Awaitable[None], None]

OnRouteHook = Callable[[ShiftRequest, RoutingDecision], _MaybeAwaitable]
OnErrorHook = Callable[[ShiftRequest, Exception], _MaybeAwaitable]
OnUsageHook = Callable[[str, int, int], _MaybeAwaitable]  # tier_name, input_tok, output_tok
OnFallbackHook = Callable[[str, str, Exception], _MaybeAwaitable]  # failed_tier, next_tier, exc


class HookRegistry:
    """Register sync or async callbacks for routing lifecycle events.

    Example::

        hooks = HookRegistry()

        @hooks.on_route
        def log_decision(request, decision):
            print(f"{decision.tier_name} selected in {decision.overhead_ms:.2f}ms")

        @hooks.on_fallback
        async def alert_fallback(failed_tier, next_tier, exc):
            await notify_slack(f"{failed_tier} failed, falling back to {next_tier}")

        router = RouterShifter(config, hooks=hooks)
    """

    __slots__ = ("_on_route", "_on_error", "_on_usage", "_on_fallback")

    def __init__(self) -> None:
        self._on_route: list[OnRouteHook] = []
        self._on_error: list[OnErrorHook] = []
        self._on_usage: list[OnUsageHook] = []
        self._on_fallback: list[OnFallbackHook] = []

    # --- decorators ---------------------------------------------------------

    def on_route(self, fn: OnRouteHook) -> OnRouteHook:
        """Fired after every successful routing decision."""
        self._on_route.append(fn)
        return fn

    def on_error(self, fn: OnErrorHook) -> OnErrorHook:
        """Fired when routing itself raises an exception."""
        self._on_error.append(fn)
        return fn

    def on_usage(self, fn: OnUsageHook) -> OnUsageHook:
        """Fired when record_usage() is called."""
        self._on_usage.append(fn)
        return fn

    def on_fallback(self, fn: OnFallbackHook) -> OnFallbackHook:
        """Fired when route_with_fallback() skips a failed tier."""
        self._on_fallback.append(fn)
        return fn

    # --- internal fire helpers ----------------------------------------------

    async def _fire(self, fns: list, *args: object) -> None:
        for fn in fns:
            result = fn(*args)
            if asyncio.iscoroutine(result):
                await result

    async def fire_route(self, request: ShiftRequest, decision: RoutingDecision) -> None:
        await self._fire(self._on_route, request, decision)

    async def fire_error(self, request: ShiftRequest, exc: Exception) -> None:
        await self._fire(self._on_error, request, exc)

    async def fire_usage(self, tier_name: str, input_tokens: int, output_tokens: int) -> None:
        await self._fire(self._on_usage, tier_name, input_tokens, output_tokens)

    async def fire_fallback(
        self, failed_tier: str, next_tier: str, exc: Exception
    ) -> None:
        await self._fire(self._on_fallback, failed_tier, next_tier, exc)
