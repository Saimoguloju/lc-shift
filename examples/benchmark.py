#!/usr/bin/env python3
"""Compare routing strategies on a labelled dataset.

    uv run python examples/benchmark.py

Prints a cost-quality table — the kind of evidence hiring managers and router
benchmarks (RouteLLM, RouterBench) actually care about: how much each strategy
saves versus always calling the strongest model, and how much quality it keeps.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from lc_shift import (
    PRESETS,
    RouterConfig,
    RouterShifter,
    Strategy,
    evaluate,
    load_dataset,
)

DATASET = Path(__file__).with_name("benchmark_dataset.jsonl")

# Derive KNN training examples and semantic routes from a few seed prompts so the
# learned strategies have something to match against.
KNN_EXAMPLES = {
    "economy": ["hello there", "what is the capital of spain", "tell me a joke", "convert units"],
    "balanced": ["write a python function and explain it", "compare two databases", "summarize a concept"],
    "performance": [
        "prove the correctness of a distributed consensus algorithm",
        "analyze and compare compiler optimization trade-offs",
        "derive the time and space complexity and justify the choice",
    ],
}


def _config(strategy: Strategy) -> RouterConfig:
    tiers = PRESETS["mixed-frontier"]
    kwargs: dict[str, object] = dict(tiers=tiers, default_tier="balanced", strategy=strategy)
    if strategy is Strategy.COMPLEXITY:
        kwargs["complexity_threshold"] = 0.35
    elif strategy is Strategy.SEMANTIC:
        kwargs["semantic_routes"] = KNN_EXAMPLES
    elif strategy is Strategy.KNN:
        kwargs["knn_examples"] = KNN_EXAMPLES
        kwargs["knn_k"] = 3
    elif strategy is Strategy.CLASSIFIER:
        kwargs["classifier_weights"] = {
            "prove": 2.5, "analyze": 2.0, "derive": 2.0, "compare": 1.2,
            "complexity": 1.8, "consensus": 2.0, "optimization": 1.5,
            "hello": -2.5, "capital": -2.0, "joke": -2.5, "translate": -2.0,
        }
        kwargs["classifier_intercept"] = -0.8
        kwargs["classifier_threshold"] = 0.6
    elif strategy is Strategy.ENSEMBLE:
        kwargs["complexity_threshold"] = 0.35
        kwargs["semantic_routes"] = KNN_EXAMPLES
        kwargs["classifier_weights"] = {"prove": 2.5, "analyze": 2.0, "hello": -2.5, "joke": -2.5}
        kwargs["ensemble_weights"] = {"complexity": 1.0, "semantic": 1.0, "classifier": 0.5}
    return RouterConfig(**kwargs)  # type: ignore[arg-type]


async def main() -> None:
    dataset = load_dataset(DATASET)
    strategies = [
        Strategy.COMPLEXITY,
        Strategy.SEMANTIC,
        Strategy.CLASSIFIER,
        Strategy.KNN,
        Strategy.ENSEMBLE,
    ]

    print(f"Benchmarking {len(strategies)} strategies on {len(dataset)} prompts "
          f"(preset: mixed-frontier)\n")
    header = f"{'strategy':<12} {'accuracy':>9} {'quality':>9} {'savings':>9} {'overhead':>11}"
    print(header)
    print("-" * len(header))

    for strategy in strategies:
        router = RouterShifter(_config(strategy))
        result = await evaluate(router, dataset)
        print(
            f"{result.strategy:<12} "
            f"{result.accuracy:>8.1%} "
            f"{result.quality_preserved:>8.1%} "
            f"{result.cost_savings_pct:>8.1f}% "
            f"{result.avg_overhead_ms:>9.4f}ms"
        )

    print("\nquality = share of prompts NOT under-routed (no quality loss)")
    print("savings = cost reduction vs. sending every prompt to the strongest tier")


if __name__ == "__main__":
    asyncio.run(main())
