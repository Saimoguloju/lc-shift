"""In-memory routing decision cache with TTL and LRU eviction."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from lc_shift.models import RoutingDecision


@dataclass
class _Entry:
    decision: RoutingDecision
    expires_at: float
    hits: int = 0


class RoutingCache:
    """Cache routing decisions for identical prompts to avoid re-scoring.

    Keyed by SHA-256(strategy + prompt), scoped so strategy changes
    don't return stale decisions.

    Example::

        cache = RoutingCache(ttl_seconds=120, max_size=2000)
        router = RouterShifter(config, cache=cache)

        # Second identical request is served from cache — zero overhead.
        d1 = await router.route(ShiftRequest(prompt="hello"))
        d2 = await router.route(ShiftRequest(prompt="hello"))
        assert d2.overhead_ms < 0.01
    """

    __slots__ = ("_ttl", "_max_size", "_store")

    def __init__(self, ttl_seconds: float = 60.0, max_size: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, _Entry] = {}

    # --- public API ---------------------------------------------------------

    def get(self, prompt: str, strategy: str) -> RoutingDecision | None:
        key = self._key(prompt, strategy)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        entry.hits += 1
        return entry.decision

    def set(self, prompt: str, strategy: str, decision: RoutingDecision) -> None:
        if len(self._store) >= self._max_size:
            self._evict()
        key = self._key(prompt, strategy)
        self._store[key] = _Entry(
            decision=decision,
            expires_at=time.monotonic() + self._ttl,
        )

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def total_hits(self) -> int:
        return sum(e.hits for e in self._store.values())

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _key(prompt: str, strategy: str) -> str:
        return hashlib.sha256(f"{strategy}:{prompt}".encode()).hexdigest()[:24]

    def _evict(self) -> None:
        now = time.monotonic()
        # Drop all expired first
        expired = [k for k, e in self._store.items() if now > e.expires_at]
        for k in expired:
            del self._store[k]
        # If still full, drop lowest-hit entry
        if len(self._store) >= self._max_size:
            lru_key = min(self._store, key=lambda k: self._store[k].hits)
            del self._store[lru_key]
