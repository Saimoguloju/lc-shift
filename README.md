# lc-shift

LLM request router that picks the right model tier for each prompt.

## Why this exists

Most LLM-powered apps use one model for everything. That's either wasteful (paying frontier prices for "what's 2+2?") or limiting (using the cheap model when you actually need deep reasoning). This is sometimes called the "Two-Language Problem" — borrowed from scientific computing where you prototype in Python but ship in C++. Same idea here: you want the *cheap fast thing* most of the time and the *expensive smart thing* when it matters.

`lc-shift` sits between your app and your providers. You define tiers (e.g. Haiku for simple stuff, Opus for hard stuff), pick a routing strategy, and the router figures out which tier to use. The routing is all CPU heuristics — no API calls, no ML models — so overhead is well under 1ms in practice.

It doesn't call any LLM APIs itself. It just tells you *which* tier to use, and you make the call with whatever SDK you're already using.

## Install

```bash
uv add lc-shift
# or
pip install lc-shift
```

## Usage

```python
import asyncio
from lc_shift import ModelTier, RouterConfig, RouterShifter, ShiftRequest, Strategy

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
        "economy": ModelTier(
            name="Economy",
            provider="anthropic",
            model_id="claude-haiku-4-5",
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
            avg_latency_ms=400,
        ),
    },
    default_tier="economy",
    strategy=Strategy.COMPLEXITY,
    complexity_threshold=0.4,
)

router = RouterShifter(config)

async def handle(prompt: str):
    decision = await router.route(ShiftRequest(prompt=prompt))
    print(f"{decision.tier_name}: {decision.reason} ({decision.overhead_ms:.2f}ms)")

    # call your provider, then record usage for cost tracking
    router.record_usage(decision.tier_name, input_tokens=200, output_tokens=500)

asyncio.run(handle("What is 2+2?"))
# economy: complexity=0.00 < threshold=0.4 (0.02ms)

asyncio.run(handle("Analyze this code and explain the trade-off between..."))
# performance: complexity=0.55 >= threshold=0.4 (0.03ms)
```

## Strategies

- **COMPLEXITY** — Scores the prompt (length, code blocks, reasoning keywords, multi-step structure). Simple stuff goes cheap, complex stuff goes premium.
- **COST_AWARE** — Uses the best tier while budget is healthy, then downshifts as you spend. At 80% consumed it drops to the cheapest tier.
- **CASCADE** — Always starts with the cheapest tier. Your app checks the response and escalates if it's not good enough.
- **LATENCY** — Picks the most capable tier that fits under your latency target.

## Cost tracking

```python
router.record_usage("economy", input_tokens=1000, output_tokens=500)
snap = router.snapshot()
# snap.total_requests, snap.estimated_cost_usd, snap.budget_remaining_usd
```

## Force a tier

```python
decision = await router.route(ShiftRequest(prompt="...", force_tier="performance"))
```

## Development

```bash
git clone https://github.com/YOUR_USERNAME/lc-shift.git
cd lc-shift
uv sync --dev
uv run pytest -v
uv run ruff check src/ tests/
uv run mypy src/
```

## What's next

Near-term stuff we're thinking about:
- Replace the regex heuristics with a small trained classifier (still CPU-only, still fast)
- Optional provider integrations (`lc-shift[anthropic]`, `lc-shift[openai]`) so you don't have to wire up the SDK calls yourself
- Quality feedback loop — if economy tier keeps producing bad results for certain prompt patterns, auto-escalate
- Multi-provider failover (Anthropic down? reroute to OpenAI)
- Semantic caching for near-duplicate prompts
- Some kind of dashboard or OpenTelemetry export for cost/routing visibility

Longer term:
- A/B routing for model evaluation
- Mid-stream rerouting during streaming responses
- Plugin system so you can write custom strategies without forking
- Multi-modal routing (images/audio have different cost profiles)

Ideas and PRs welcome.

## License

MIT
