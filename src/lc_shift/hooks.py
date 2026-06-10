from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Union

from lc_shift.models import RoutingDecision, ShiftRequest

_MaybeAwaitable = Union[Awaitable[None], None]

OnRouteHook = Callable[[ShiftRequest, RoutingDecision], _MaybeAwaitable]
OnErrorHook = Callable[[ShiftRequest, Exception], _MaybeAwaitable]
OnUsageHook = Callable[[str, int, int], _MaybeAwaitable]
OnFallbackHook = Callable[[str, str, Exception], _MaybeAwaitable]


class HookRegistry:
    """Registry for routing lifecycle hook callbacks."""

    __slots__ = ("_on_route", "_on_error", "_on_usage", "_on_fallback")

    def __init__(self) -> None:
        self._on_route: list[OnRouteHook] = []
        self._on_error: list[OnErrorHook] = []
        self._on_usage: list[OnUsageHook] = []
        self._on_fallback: list[OnFallbackHook] = []

    def on_route(self, fn: OnRouteHook) -> OnRouteHook:
        self._on_route.append(fn)
        return fn

    def on_error(self, fn: OnErrorHook) -> OnErrorHook:
        self._on_error.append(fn)
        return fn

    def on_usage(self, fn: OnUsageHook) -> OnUsageHook:
        self._on_usage.append(fn)
        return fn

    def on_fallback(self, fn: OnFallbackHook) -> OnFallbackHook:
        self._on_fallback.append(fn)
        return fn

    async def _fire(self, fns: list[Callable[..., Any]], *args: Any) -> None:
        for fn in fns:
            result = fn(*args)
            if asyncio.iscoroutine(result):
                await result

    @staticmethod
    def _dispatch(fns: list[Callable[..., Any]], *args: Any) -> None:
        """Fire hooks from a synchronous context.

        Sync callbacks run inline. Async callbacks are scheduled on the running
        loop if one exists; otherwise they are executed to completion. This keeps
        ``record_usage`` usable from both sync and async call sites without
        relying on the deprecated ``get_event_loop()``.
        """
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for fn in fns:
            result = fn(*args)
            if asyncio.iscoroutine(result):
                if loop is not None:
                    loop.create_task(result)
                else:
                    asyncio.run(result)

    async def fire_route(self, request: ShiftRequest, decision: RoutingDecision) -> None:
        await self._fire(self._on_route, request, decision)

    async def fire_error(self, request: ShiftRequest, exc: Exception) -> None:
        await self._fire(self._on_error, request, exc)

    async def fire_usage(self, tier_name: str, input_tokens: int, output_tokens: int) -> None:
        await self._fire(self._on_usage, tier_name, input_tokens, output_tokens)

    def dispatch_usage(self, tier_name: str, input_tokens: int, output_tokens: int) -> None:
        """Synchronous entry point for usage hooks (see :meth:`_dispatch`)."""
        self._dispatch(self._on_usage, tier_name, input_tokens, output_tokens)

    async def fire_fallback(
        self, failed_tier: str, next_tier: str, exc: Exception
    ) -> None:
        await self._fire(self._on_fallback, failed_tier, next_tier, exc)
