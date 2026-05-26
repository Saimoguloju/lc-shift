from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TierHealth:
    """Manages cooldowns for degraded LLM tiers to avoid calling down providers."""

    cooldown_seconds: float = 60.0
    _failing: dict[str, float] = field(default_factory=dict, repr=False)

    def mark_failed(self, tier_name: str) -> None:
        self._failing[tier_name] = time.monotonic() + self.cooldown_seconds

    def recover(self, tier_name: str) -> None:
        self._failing.pop(tier_name, None)

    def is_healthy(self, tier_name: str) -> bool:
        expires = self._failing.get(tier_name)
        if expires is None:
            return True
        if time.monotonic() > expires:
            self._failing.pop(tier_name, None)
            return True
        return False

    def seconds_until_recovery(self, tier_name: str) -> float:
        expires = self._failing.get(tier_name)
        if expires is None:
            return 0.0
        return max(0.0, expires - time.monotonic())

    @property
    def degraded_tiers(self) -> list[str]:
        now = time.monotonic()
        return [t for t, exp in self._failing.items() if now < exp]

    @property
    def all_healthy(self) -> bool:
        return len(self.degraded_tiers) == 0
