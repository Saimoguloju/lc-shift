#!/usr/bin/env python3
"""uv run python examples/basic_usage.py"""

from __future__ import annotations

import asyncio

from lc_shift import (
    RouterConfig,
    RouterShifter,
    ShiftRequest,
    Strategy,
    HookRegistry,
    RoutingCache,
    TierHealth,
    PRESETS,
    ANTHROPIC,
    OPENAI,
    GOOGLE,
    DEEPSEEK,
    GROQ,
    OLLAMA,
)


# ---------------------------------------------------------------------------
# Example 1 — Async context manager + all-Anthropic tiers
# ---------------------------------------------------------------------------
async def example_context_manager() -> None:
    print("=" * 60)
    print("Example 1: Async context manager + Anthropic tiers")
    print("=" * 60)

    config = RouterConfig(
        tiers=PRESETS["anthropic-3tier"],
        default_tier="balanced",
        strategy=Strategy.COMPLEXITY,
        complexity_threshold=0.4,
    )

    async with RouterShifter(config) as router:
        prompts = [
            "What is 2 + 2?",
            (
                "Analyze the following code and explain why the recursive approach "
                "has worse performance. Compare time and space complexity trade-offs.\n"
                "```python\ndef fib(n):\n    return fib(n-1)+fib(n-2) if n>1 else n\n```\n"
                "1. Evaluate time complexity\n"
                "2. Assess space complexity\n"
                "3. Justify production choice"
            ),
            "Translate 'hello' to Spanish.",
        ]

        for prompt in prompts:
            decision = await router.route(ShiftRequest(prompt=prompt))
            print(f"Prompt:    {prompt[:55]}...")
            print(f"  Tier:    {decision.tier_name} ({decision.tier.model_id})")
            print(f"  Why:     {decision.reason}")
            print(f"  Time:    {decision.overhead_ms:.3f} ms | Cache hit: {decision.cache_hit}")
            router.record_usage(decision.tier_name, input_tokens=150, output_tokens=300)

    snap = router.snapshot()
    print(f"\nCost: ${snap.estimated_cost_usd:.6f} across {snap.total_requests} requests\n")


# ---------------------------------------------------------------------------
# Example 2 — Observability hooks
# ---------------------------------------------------------------------------
async def example_hooks() -> None:
    print("=" * 60)
    print("Example 2: Observability hooks")
    print("=" * 60)

    hooks = HookRegistry()

    @hooks.on_route
    def log_route(request, decision):
        tag = "[CACHE]" if decision.cache_hit else f"[{decision.overhead_ms:.2f}ms]"
        print(f"  HOOK on_route  {tag} -> {decision.tier_name}: {decision.reason}")

    @hooks.on_usage
    def log_usage(tier_name, input_tokens, output_tokens):
        print(f"  HOOK on_usage  {tier_name}: {input_tokens}in / {output_tokens}out tokens")

    @hooks.on_fallback
    async def log_fallback(failed_tier, next_tier, exc):
        print(f"  HOOK on_fallback  {failed_tier} failed -> trying {next_tier}")

    config = RouterConfig(
        tiers=PRESETS["mixed-frontier"],
        default_tier="balanced",
        strategy=Strategy.COMPLEXITY,
    )
    router = RouterShifter(config, hooks=hooks)

    for prompt in ["Quick summary", "Deep multi-step reasoning analysis with code review"]:
        decision = await router.route(ShiftRequest(prompt=prompt))
        router.record_usage(decision.tier_name, input_tokens=100, output_tokens=200)

    print()


# ---------------------------------------------------------------------------
# Example 3 — Routing cache (identical prompts served from cache)
# ---------------------------------------------------------------------------
async def example_cache() -> None:
    print("=" * 60)
    print("Example 3: Routing decision cache (TTL=30s)")
    print("=" * 60)

    cache = RoutingCache(ttl_seconds=30, max_size=500)
    config = RouterConfig(
        tiers=PRESETS["anthropic-3tier"],
        default_tier="balanced",
        strategy=Strategy.COMPLEXITY,
    )
    router = RouterShifter(config, cache=cache)

    prompt = "Explain transformers in NLP in detail with mathematical derivations."

    d1 = await router.route(ShiftRequest(prompt=prompt))
    print(f"  1st call -> {d1.tier_name} | overhead: {d1.overhead_ms:.3f}ms | cache: {d1.cache_hit}")

    d2 = await router.route(ShiftRequest(prompt=prompt))
    print(f"  2nd call -> {d2.tier_name} | overhead: {d2.overhead_ms:.3f}ms | cache: {d2.cache_hit}")

    print(f"  Cache size: {cache.size} | total hits: {cache.total_hits}\n")


# ---------------------------------------------------------------------------
# Example 4 — Fallback chain with tier health tracking
# ---------------------------------------------------------------------------
async def example_fallback_and_health() -> None:
    print("=" * 60)
    print("Example 4: Fallback chain + tier health tracking")
    print("=" * 60)

    health = TierHealth(cooldown_seconds=30.0)
    config = RouterConfig(
        tiers={
            "performance": ANTHROPIC["claude-opus-4-6"],
            "balanced":    OPENAI["gpt-4o"],
            "economy":     GOOGLE["gemini-flash"],
        },
        default_tier="balanced",
        strategy=Strategy.COMPLEXITY,
        complexity_threshold=0.3,
    )
    router = RouterShifter(config, health=health)

    request = ShiftRequest(prompt="Analyze and compare distributed consensus algorithms.")

    # Simulate: primary provider (performance) goes down
    router.mark_tier_failed("performance")
    print(f"  Degraded tiers: {router.health.degraded_tiers}")

    chain = await router.route_with_fallback(request)
    print(f"  Fallback chain ({len(chain)} tiers):")
    for i, decision in enumerate(chain):
        label = "PRIMARY" if i == 0 else f"FALLBACK {i}"
        print(f"    [{label}] {decision.tier_name} ({decision.tier.provider}/{decision.tier.model_id})")
    print(f"  Skipped (degraded): {chain.skipped_tiers}\n")

    # Use primary decision from chain
    best = chain.primary
    router.record_usage(best.tier_name, input_tokens=300, output_tokens=600)


# ---------------------------------------------------------------------------
# Example 5 — Batch routing (concurrent)
# ---------------------------------------------------------------------------
async def example_batch() -> None:
    print("=" * 60)
    print("Example 5: Batch routing (concurrent)")
    print("=" * 60)

    config = RouterConfig(
        tiers={
            "fast":    GROQ["llama3-groq"],
            "smart":   DEEPSEEK["deepseek-v3.2"],
            "local":   OLLAMA["llama3.3"],
        },
        default_tier="fast",
        strategy=Strategy.LATENCY,
        latency_target_ms=300,
    )
    router = RouterShifter(config)

    requests = [
        ShiftRequest(prompt="What is Python?"),
        ShiftRequest(prompt="Write a distributed rate limiter with Redis. Include code, explain trade-offs."),
        ShiftRequest(prompt="Capital of France?"),
        ShiftRequest(prompt="Explain gradient descent with math derivation, compare Adam vs SGD."),
    ]

    import time
    t0 = time.perf_counter()
    decisions = await router.route_batch(requests)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Routed {len(decisions)} requests in {elapsed:.2f}ms total")
    for req, dec in zip(requests, decisions):
        print(f"  [{dec.tier_name:8}] {req.prompt[:55]}...")
    print()


# ---------------------------------------------------------------------------
# Example 6 — Cost-aware with budget controls
# ---------------------------------------------------------------------------
async def example_budget() -> None:
    print("=" * 60)
    print("Example 6: Cost-aware routing with $0.50 budget")
    print("=" * 60)

    config = RouterConfig(
        tiers=PRESETS["mixed-frontier"],
        default_tier="balanced",
        strategy=Strategy.COST_AWARE,
        cost_budget_usd=0.50,
    )
    router = RouterShifter(config)

    stages = [
        ("Early (budget healthy)", 0),
        ("Mid (50% spent)", 250),
        ("Late (80% spent)", 150),
    ]

    for label, extra_tokens in stages:
        if extra_tokens:
            router.record_usage("performance", input_tokens=extra_tokens * 10, output_tokens=extra_tokens * 20)
        decision = await router.route(ShiftRequest(prompt="Process this task."))
        snap = router.snapshot()
        pct = (snap.estimated_cost_usd / 0.50) * 100
        print(f"  {label}: -> {decision.tier_name} | spent: ${snap.estimated_cost_usd:.4f} ({pct:.0f}%)")

    print()


# ---------------------------------------------------------------------------
# Example 7 — List all 35 providers
# ---------------------------------------------------------------------------
def example_list_providers() -> None:
    print("=" * 60)
    print("Example 7: All 35 pre-configured provider presets")
    print("=" * 60)

    from lc_shift import ALL_PROVIDERS
    print(f"  Total models: {len(ALL_PROVIDERS)}\n")
    for key, tier in ALL_PROVIDERS.items():
        cost = f"${tier.cost_per_1k_input:.5f}/${tier.cost_per_1k_output:.5f}"
        tag = " [LOCAL/FREE]" if tier.cost_per_1k_input == 0.0 else ""
        print(f"  {key:<52} {cost}{tag}")

    print(f"\n  PRESETS: {list(PRESETS.keys())}\n")


# ---------------------------------------------------------------------------
# Run all examples
# ---------------------------------------------------------------------------
async def main() -> None:
    await example_context_manager()
    await example_hooks()
    await example_cache()
    await example_fallback_and_health()
    await example_batch()
    await example_budget()
    example_list_providers()


if __name__ == "__main__":
    asyncio.run(main())
