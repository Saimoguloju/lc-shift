"""Agentic routing helpers.

Agent loops are the ideal use case for a router: a single task fans out into many
LLM calls (plan, select tool, reflect, summarize) of wildly different difficulty.
This module adds two agent-native primitives on top of :class:`RouterShifter`:

* **Role routing** — pick a tier per agent step via ``metadata['role']`` (see
  :class:`~lc_shift.strategies.RoleStrategy`).
* **Escalation** — start on the cheapest tier and climb to a stronger one only
  when a validation check fails, the canonical "cheap-first, escalate-on-failure"
  agent pattern. This is model-agnostic: you provide the call/validate callbacks,
  lc-shift decides *which* model each attempt uses.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Union

from pydantic import BaseModel

from lc_shift.config import ModelTier
from lc_shift.models import RoutingDecision, ShiftRequest
from lc_shift.router import RouterShifter

# A model-call callback: given the chosen tier and prompt, return output text.
CallFn = Callable[[ModelTier, str], Union[str, Awaitable[str]]]
# A validator: return True if the output is acceptable (no escalation needed).
ValidateFn = Callable[[str], bool]


class EscalationAttempt(BaseModel):
    tier_name: str
    model_id: str
    output: str
    accepted: bool


class EscalationResult(BaseModel):
    output: str
    tier_name: str
    model_id: str
    success: bool
    escalated: bool
    attempts: list[EscalationAttempt]


class AgentRouter:
    """Thin agent-focused wrapper around a :class:`RouterShifter`."""

    __slots__ = ("_router",)

    def __init__(self, router: RouterShifter) -> None:
        self._router = router

    @property
    def router(self) -> RouterShifter:
        return self._router

    def escalation_order(self, *, start_tier: str | None = None) -> list[str]:
        """Tier names ordered cheapest -> strongest (by input cost).

        If ``start_tier`` is given, the order begins at that tier (skipping any
        cheaper ones).
        """
        tiers = self._router.config.tiers
        ordered = [name for name, _ in sorted(tiers.items(), key=lambda kv: kv[1].cost_per_1k_input)]
        if start_tier is not None:
            if start_tier not in tiers:
                raise ValueError(f"start_tier '{start_tier}' not in tiers: {list(tiers)}")
            ordered = ordered[ordered.index(start_tier):]
        return ordered

    async def route_step(
        self,
        prompt: str,
        *,
        role: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> RoutingDecision:
        """Route a single agent step, attaching an optional ``role`` tag."""
        meta = dict(metadata or {})
        if role is not None:
            meta["role"] = role
        return await self._router.route(ShiftRequest(prompt=prompt, metadata=meta))

    async def run_with_escalation(
        self,
        prompt: str,
        call: CallFn,
        *,
        validate: ValidateFn | None = None,
        start_tier: str | None = None,
        max_attempts: int | None = None,
    ) -> EscalationResult:
        """Run ``call`` against progressively stronger tiers until it passes.

        Walks the escalation order (cheapest first). For each tier it invokes
        ``call(tier, prompt)`` and checks ``validate(output)``; the first accepted
        output wins. If none pass, the last (strongest attempted) output is
        returned with ``success=False``. ``call`` may be sync or async.
        """
        order = self.escalation_order(start_tier=start_tier)
        if max_attempts is not None:
            order = order[:max_attempts]

        tiers = self._router.config.tiers
        attempts: list[EscalationAttempt] = []

        for tier_name in order:
            tier = tiers[tier_name]
            result = call(tier, prompt)
            output = await result if inspect.isawaitable(result) else result
            output = str(output)
            accepted = validate is None or validate(output)
            attempts.append(
                EscalationAttempt(
                    tier_name=tier_name,
                    model_id=tier.model_id,
                    output=output,
                    accepted=accepted,
                )
            )
            if accepted:
                return EscalationResult(
                    output=output,
                    tier_name=tier_name,
                    model_id=tier.model_id,
                    success=True,
                    escalated=len(attempts) > 1,
                    attempts=attempts,
                )

        last = attempts[-1]
        return EscalationResult(
            output=last.output,
            tier_name=last.tier_name,
            model_id=last.model_id,
            success=False,
            escalated=len(attempts) > 1,
            attempts=attempts,
        )
