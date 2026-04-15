from __future__ import annotations

import abc
import re

from lc_shift.config import RouterConfig
from lc_shift.models import ShiftRequest

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`]+`")

_REASONING_KEYWORDS = frozenset({
    "analyze", "compare", "contrast", "explain", "evaluate",
    "synthesize", "critique", "assess", "justify", "derive",
    "prove", "reason", "why", "how does", "what if",
    "trade-off", "implications",
})

_MULTI_PART_RE = re.compile(
    r"\b\d+[.\)]\s|\b(?:first|second|third|finally|additionally)\b",
    re.IGNORECASE,
)


def estimate_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def compute_complexity(prompt: str) -> float:
    """Score from 0-1 based on length, code, keywords, and structure."""
    score = 0.0
    tokens = estimate_token_count(prompt)
    lower = prompt.lower()

    if tokens > 500:
        score += 0.3
    elif tokens > 200:
        score += 0.2
    elif tokens > 50:
        score += 0.1

    if _CODE_BLOCK_RE.search(prompt):
        score += 0.25

    keyword_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in lower)
    score += min(0.25, keyword_hits * 0.05)

    parts = len(_MULTI_PART_RE.findall(prompt))
    score += min(0.2, parts * 0.05)

    return min(1.0, score)


class BaseStrategy(abc.ABC):
    @abc.abstractmethod
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        """Return (tier_name, reason)."""
        ...


class ComplexityStrategy(BaseStrategy):
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        score = compute_complexity(request.prompt)
        tiers_by_cost = sorted(
            config.tiers.items(),
            key=lambda t: t[1].cost_per_1k_input,
        )
        if score >= config.complexity_threshold:
            name = tiers_by_cost[-1][0]
            reason = f"complexity={score:.2f} >= threshold={config.complexity_threshold}"
        else:
            name = tiers_by_cost[0][0]
            reason = f"complexity={score:.2f} < threshold={config.complexity_threshold}"
        return name, reason


class CostAwareStrategy(BaseStrategy):
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        budget = config.cost_budget_usd
        tiers_by_cost = sorted(
            config.tiers.items(),
            key=lambda t: t[1].cost_per_1k_input,
        )

        if budget is not None:
            remaining = budget - spent_usd
            if remaining <= 0:
                return tiers_by_cost[0][0], f"budget exhausted (spent=${spent_usd:.4f})"

            fraction_used = spent_usd / budget if budget > 0 else 1.0
            if fraction_used < 0.5:
                return tiers_by_cost[-1][0], f"budget healthy ({fraction_used:.0%} used)"
            elif fraction_used < 0.8:
                mid = len(tiers_by_cost) // 2
                return tiers_by_cost[mid][0], f"budget moderate ({fraction_used:.0%} used)"
            else:
                return tiers_by_cost[0][0], f"budget tight ({fraction_used:.0%} used)"

        return config.default_tier, "no budget constraint set"


class CascadeStrategy(BaseStrategy):
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        cheapest = min(config.tiers.items(), key=lambda t: t[1].cost_per_1k_input)
        return cheapest[0], "cascade: start with cheapest tier, escalate on insufficient quality"


class LatencyStrategy(BaseStrategy):
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        target = config.latency_target_ms
        if target is None:
            return config.default_tier, "no latency target set, using default"

        eligible = [
            (name, tier)
            for name, tier in config.tiers.items()
            if tier.avg_latency_ms <= target
        ]

        if eligible:
            best = max(eligible, key=lambda t: t[1].cost_per_1k_input)
            return best[0], f"best tier meeting latency target {target}ms"

        fastest = min(config.tiers.items(), key=lambda t: t[1].avg_latency_ms)
        return fastest[0], (
            f"no tier meets {target}ms target; using fastest "
            f"({fastest[1].avg_latency_ms}ms)"
        )


STRATEGY_MAP: dict[str, type[BaseStrategy]] = {
    "complexity": ComplexityStrategy,
    "cost_aware": CostAwareStrategy,
    "cascade": CascadeStrategy,
    "latency": LatencyStrategy,
}
