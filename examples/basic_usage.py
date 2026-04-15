#!/usr/bin/env python3
"""uv run python examples/basic_usage.py"""

from __future__ import annotations

import asyncio

from lc_shift import ModelTier, RouterConfig, RouterShifter, ShiftRequest, Strategy


async def main() -> None:
    config = RouterConfig(
        tiers={
            "performance": ModelTier(
                name="Performance",
                provider="anthropic",
                model_id="claude-opus-4-6",
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                avg_latency_ms=2500,
            ),
            "balanced": ModelTier(
                name="Balanced",
                provider="anthropic",
                model_id="claude-sonnet-4-6",
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                avg_latency_ms=1200,
            ),
            "economy": ModelTier(
                name="Economy",
                provider="anthropic",
                model_id="claude-haiku-4-5",
                cost_per_1k_input=0.0008,
                cost_per_1k_output=0.004,
                avg_latency_ms=400,
            ),
        },
        default_tier="balanced",
        strategy=Strategy.COMPLEXITY,
        complexity_threshold=0.4,
    )

    router = RouterShifter(config)

    prompts = [
        "What is 2 + 2?",
        (
            "Analyze the following code and explain why the recursive approach "
            "has worse performance than the iterative one. Compare the trade-off "
            "between readability and efficiency.\n"
            "```python\n"
            "def fib_recursive(n):\n"
            "    if n <= 1: return n\n"
            "    return fib_recursive(n-1) + fib_recursive(n-2)\n"
            "```\n"
            "1. First, evaluate time complexity\n"
            "2. Then, assess space complexity\n"
            "3. Finally, justify which to use in production"
        ),
        "Translate 'hello' to Spanish.",
    ]

    for prompt in prompts:
        decision = await router.route(ShiftRequest(prompt=prompt))

        print(f"Prompt:    {prompt[:60]}...")
        print(f"  -> Tier: {decision.tier_name} ({decision.tier.model_id})")
        print(f"  -> Why:  {decision.reason}")
        print(f"  -> Time: {decision.overhead_ms:.3f} ms")
        print()

        # you'd call your actual provider here, then record usage
        router.record_usage(decision.tier_name, input_tokens=150, output_tokens=300)

    snap = router.snapshot()
    print("Cost snapshot:")
    print(f"  requests:  {snap.total_requests}")
    print(f"  cost:      ${snap.estimated_cost_usd:.6f}")
    print(f"  by tier:   {snap.requests_by_tier}")


if __name__ == "__main__":
    asyncio.run(main())
