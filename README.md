# lc-shift

A provider-agnostic LLM router that dynamically determines the optimal model tier for each prompt in under 1ms without network calls or ML models.

`lc-shift` sits between your application logic and your LLM providers. By defining custom tiers (e.g., Performance, Balanced, Economy) and choosing a routing strategy, it evaluates incoming prompts using local CPU-based heuristics to decide which model tier is most appropriate.

---

## Installation

```bash
pip install lc-shift
```

**Requirements:** Python 3.11+ and Pydantic v2.

---

## Quick Start

```python
import asyncio
from lc_shift import RouterShifter, RouterConfig, ShiftRequest, Strategy, PRESETS

config = RouterConfig(
    tiers=PRESETS["anthropic-3tier"],
    default_tier="balanced",
    strategy=Strategy.COMPLEXITY,
    complexity_threshold=0.4,
)

async def main():
    async with RouterShifter(config) as router:
        # Route a simple request
        decision = await router.route(ShiftRequest(prompt="What is 2+2?"))
        print(f"Tier: {decision.tier_name} ({decision.reason})")
        
        # Log usage to track cost budget
        router.record_usage(decision.tier_name, input_tokens=20, output_tokens=5)

asyncio.run(main())
```

---

## Routing Strategies

| Strategy | Goal | Key Parameters |
|---|---|---|
| `COMPLEXITY` | Evaluates prompt complexity (length, syntax, code blocks) to route simple tasks to cheap tiers and complex queries to frontier tiers. | `complexity_threshold` |
| `COST_AWARE` | Maximizes performance under a set budget, dynamically downgrading tiers as consumption approaches the limit. | `cost_budget_usd` |
| `CASCADE` | Starts with the cheapest tier. Ideal when your app evaluates outputs and manually escalates on failure. | - |
| `LATENCY` | Filters tiers meeting a latency limit, routing to the most capable tier within target. | `latency_target_ms` |

---

## Features & Usage

### 1. Pre-Configured Presets
Quickly leverage built-in model definitions and pricing profiles:
```python
from lc_shift import PRESETS, RouterConfig

config = RouterConfig(
    tiers=PRESETS["mixed-frontier"],  # Claude Opus / GPT-4o / Gemini Flash
    default_tier="balanced"
)
```

### 2. Fallbacks & Health Degradation
Exclude failing or throttled providers from the routing pool automatically:
```python
from lc_shift import RouterShifter, TierHealth, ShiftRequest

health = TierHealth(cooldown_seconds=60)
router = RouterShifter(config, health=health)

chain = await router.route_with_fallback(ShiftRequest(prompt="hello"))
for decision in chain:
    try:
        # Call LLM client here using decision.tier
        break
    except Exception:
        router.mark_tier_failed(decision.tier_name)
```

### 3. Pluggable Observability Hooks
Register event-driven callbacks for monitoring, logging, and error tracing:
```python
from lc_shift import HookRegistry, RouterShifter

hooks = HookRegistry()

@hooks.on_route
def log_routing(request, decision):
    print(f"Routed to {decision.tier_name} in {decision.overhead_ms:.2f}ms")

router = RouterShifter(config, hooks=hooks)
```

### 4. Routing Decision Cache
Avoid re-scoring identical prompts using the built-in TTL cache:
```python
from lc_shift import RoutingCache, RouterShifter

cache = RoutingCache(ttl_seconds=60, max_size=1000)
router = RouterShifter(config, cache=cache)
```

---

## Development

Set up your development environment using `uv`:

```bash
git clone https://github.com/Saimoguloju/lc-shift.git
cd lc-shift
uv sync --dev

# Run tests
uv run pytest

# Lint and formatting check
uv run ruff check src/ tests/
uv run mypy src/
```

---

## License

MIT
