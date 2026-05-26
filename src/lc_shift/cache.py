from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from lc_shift.models import RoutingDecision


@dataclass
class _Entry:
    decision: RoutingDecision
    expires_at: float
    hits: int = 0


class RoutingCache:
    """In-memory cache for routing decisions.
    
    Keys are generated using a combination of the prompt and routing strategy
    to ensure changes in configuration correctly bust the cache.
    """

    __slots__ = ("_ttl", "_max_size", "_store")

    def __init__(self, ttl_seconds: float = 60.0, max_size: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, _Entry] = {}

    def get(self, prompt: str, strategy: str) -> RoutingDecision | None:
        key = self._key(prompt, strategy)
        entry = self._store.get(key)
        if not entry:
            return None
        if time.monotonic() > entry.expires_at:
            self._store.pop(key, None)
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

    @staticmethod
    def _key(prompt: str, strategy: str) -> str:
        data = f"{strategy}:{prompt}".encode()
        return hashlib.sha256(data).hexdigest()[:24]

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, e in self._store.items() if now > e.expires_at]
        for k in expired:
            self._store.pop(k, None)

        if len(self._store) >= self._max_size:
            lru_key = min(self._store, key=lambda k: self._store[k].hits)
            self._store.pop(lru_key, None)
