"""Tier health tracking — mark tiers as degraded and auto-recover after cooldown."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TierHealth:
    """Track which tiers are currently healthy and skip degraded ones.

    When your application catches a provider error (timeout, 429, 5xx),
    call ``mark_failed(tier_name)`` so the router skips that tier for
    ``cooldown_seconds`` before retrying it automatically.

    Example::

        router = RouterShifter(config)

        try:
            result = await call_llm(decision.tier)
        except ProviderError as e:
            router.mark_tier_failed(decision.tier_name)
            fallback = await router.route_with_fallback(request)
            # fallback is the next healthy tier
    """

    cooldown_seconds: float = 60.0
    _failing: dict[str, float] = field(default_factory=dict, repr=False)

    def mark_failed(self, tier_name: str) -> None:
        """Mark tier as degraded for ``cooldown_seconds``."""
        self._failing[tier_name] = time.monotonic() + self.cooldown_seconds

    def recover(self, tier_name: str) -> None:
        """Manually clear a tier's degraded state."""
        self._failing.pop(tier_name, None)

    def is_healthy(self, tier_name: str) -> bool:
        """Return True if the tier is currently usable."""
        expires = self._failing.get(tier_name)
        if expires is None:
            return True
        if time.monotonic() > expires:
            del self._failing[tier_name]
            return True
        return False

    def seconds_until_recovery(self, tier_name: str) -> float:
        """Return seconds remaining in cooldown, or 0.0 if healthy."""
        expires = self._failing.get(tier_name)
        if expires is None:
            return 0.0
        remaining = expires - time.monotonic()
        return max(0.0, remaining)

    @property
    def degraded_tiers(self) -> list[str]:
        """Names of tiers currently in cooldown."""
        now = time.monotonic()
        return [t for t, exp in self._failing.items() if now < exp]

    @property
    def all_healthy(self) -> bool:
        """True when no tiers are in cooldown."""
        return len(self.degraded_tiers) == 0
