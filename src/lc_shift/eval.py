"""Offline evaluation harness for routing strategies.

Serious LLM routers (RouteLLM, RouterBench, RouterArena) are judged not by
features but by a **cost-quality trade-off**: how much money does the router
save versus always calling the strongest model, and how much quality does it
preserve while doing so. This module computes exactly those metrics over a
labelled dataset, with zero external dependencies.

Dataset format — JSONL, one record per line::

    {"prompt": "What is 2 + 2?", "ideal_tier": "economy"}
    {"prompt": "Prove the CAP theorem and...", "ideal_tier": "performance"}

``ideal_tier`` is the cheapest tier that can answer the prompt acceptably.
Routing *cheaper* than that is an under-route (quality risk); routing *more
expensive* is an over-route (wasted cost).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from lc_shift.config import ModelTier
from lc_shift.models import ShiftRequest
from lc_shift.router import RouterShifter


class EvalRecord(BaseModel):
    prompt: str = Field(min_length=1)
    ideal_tier: str


class BenchmarkResult(BaseModel):
    """Aggregate metrics from an evaluation run."""

    strategy: str
    total: int
    accuracy: float
    under_route_rate: float
    over_route_rate: float
    quality_preserved: float
    cost_total_usd: float
    cost_all_strong_usd: float
    cost_savings_pct: float
    avg_overhead_ms: float
    routed_distribution: dict[str, int] = Field(default_factory=dict)

    def format_report(self) -> str:
        """Human-readable summary suitable for a CLI or README badge."""
        lines = [
            f"Strategy:           {self.strategy}",
            f"Samples:            {self.total}",
            f"Routing accuracy:   {self.accuracy:.1%}",
            f"Quality preserved:  {self.quality_preserved:.1%}  "
            f"(under-route {self.under_route_rate:.1%}, over-route {self.over_route_rate:.1%})",
            f"Cost vs all-strong: ${self.cost_total_usd:.4f} of ${self.cost_all_strong_usd:.4f}",
            f"Cost savings:       {self.cost_savings_pct:.1f}%",
            f"Avg overhead:       {self.avg_overhead_ms:.4f} ms",
            f"Routed to:          {self.routed_distribution}",
        ]
        return "\n".join(lines)


def load_dataset(path: str | Path) -> list[EvalRecord]:
    """Load a JSONL evaluation dataset."""
    records: list[EvalRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(EvalRecord.model_validate_json(line))
    return records


def _capability_rank(tiers: dict[str, ModelTier]) -> dict[str, int]:
    """Rank tiers by cost (cheapest = 0); higher cost == more capable."""
    ordered = sorted(tiers.items(), key=lambda kv: kv[1].cost_per_1k_input)
    return {name: rank for rank, (name, _) in enumerate(ordered)}


async def evaluate(
    router: RouterShifter,
    dataset: list[EvalRecord],
    *,
    assumed_input_tokens: int = 500,
    assumed_output_tokens: int = 500,
) -> BenchmarkResult:
    """Route every record and score the router against the dataset labels.

    A fixed token budget per prompt is assumed so cost comparisons are apples to
    apples across strategies. Pass a freshly constructed ``router`` (state such
    as spend is not reset here).
    """
    if not dataset:
        raise ValueError("dataset is empty")

    tiers = router.config.tiers
    rank = _capability_rank(tiers)
    strongest = max(tiers.items(), key=lambda kv: kv[1].cost_per_1k_input)[1]

    def _cost(tier: ModelTier) -> float:
        return (
            (assumed_input_tokens / 1000) * tier.cost_per_1k_input
            + (assumed_output_tokens / 1000) * tier.cost_per_1k_output
        )

    correct = under = over = 0
    cost_total = 0.0
    overhead_total = 0.0
    distribution: dict[str, int] = {name: 0 for name in tiers}

    for record in dataset:
        if record.ideal_tier not in rank:
            raise ValueError(
                f"ideal_tier '{record.ideal_tier}' not in router tiers: {list(tiers)}"
            )
        decision = await router.route(ShiftRequest(prompt=record.prompt))
        routed = decision.tier_name
        distribution[routed] += 1
        overhead_total += decision.overhead_ms
        cost_total += _cost(tiers[routed])

        if routed == record.ideal_tier:
            correct += 1
        elif rank[routed] < rank[record.ideal_tier]:
            under += 1
        else:
            over += 1

    n = len(dataset)
    cost_all_strong = _cost(strongest) * n
    savings = (1 - cost_total / cost_all_strong) * 100 if cost_all_strong > 0 else 0.0

    return BenchmarkResult(
        strategy=router.config.strategy.value,
        total=n,
        accuracy=correct / n,
        under_route_rate=under / n,
        over_route_rate=over / n,
        quality_preserved=1 - under / n,
        cost_total_usd=round(cost_total, 6),
        cost_all_strong_usd=round(cost_all_strong, 6),
        cost_savings_pct=round(savings, 2),
        avg_overhead_ms=round(overhead_total / n, 6),
        routed_distribution=distribution,
    )
