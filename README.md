# lc-shift

**Provider-agnostic LLM router.** Picks the right model tier for every prompt — under 1ms, no API calls, no ML models required.

[![PyPI version](https://img.shields.io/pypi/v/lc-shift.svg)](https://pypi.org/project/lc-shift/)
[![Python](https://img.shields.io/pypi/pyversions/lc-shift.svg)](https://pypi.org/project/lc-shift/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/Saimoguloju/lc-shift/actions/workflows/ci.yml/badge.svg)](https://github.com/Saimoguloju/lc-shift/actions)

---

## Why lc-shift?

Most LLM apps use one model for everything — which is either wasteful (paying Opus prices for "what's 2+2?") or limiting (using the cheap model when you need deep reasoning).

`lc-shift` sits between your app and your providers. You define tiers, pick a routing strategy, and the router decides which tier each prompt deserves. Routing decisions are pure CPU heuristics — **no API calls, no ML models, <1ms overhead**.

It does **not** call any LLM APIs. It tells you *which* tier to use, and you make the call with whatever SDK you're already using — OpenAI, Anthropic, Groq, Ollama, anything.

---

## Install

```bash
pip install lc-shift
# or
uv add lc-shift
```

**Requirements:** Python 3.11+ · Pydantic v2 (only dependency)

---

## Quick start

```python
import asyncio
from lc_shift import RouterShifter, RouterConfig, ShiftRequest, Strategy, PRESETS

config = RouterConfig(
    tiers=PRESETS["anthropic-3tier"],   # Claude Opus / Sonnet / Haiku
    default_tier="balanced",
    strategy=Strategy.COMPLEXITY,
    complexity_threshold=0.4,
)

async def main():
    async with RouterShifter(config) as router:
        decision = await router.route(ShiftRequest(prompt="What is 2+2?"))
        print(f"{decision.tier_name}: {decision.reason} ({decision.overhead_ms:.2f}ms)")
        # economy: complexity=0.00 < threshold=0.4 (0.02ms)

        result = await call_your_llm(decision.tier)   # your code here
        router.record_usage(decision.tier_name, input_tokens=20, output_tokens=5)

asyncio.run(main())
```

---

## Routing strategies

| Strategy | When to use |
|---|---|
| `COMPLEXITY` | Score each prompt 0–1 (length + code + reasoning keywords + multi-step structure). Simple → cheap, complex → premium. |
| `COST_AWARE` | Use the best tier while budget is healthy, downshift as spend grows. At 80% consumed → cheapest. |
| `CASCADE` | Always start with the cheapest tier. Your app checks quality and escalates if needed. |
| `LATENCY` | Pick the most capable tier that fits under your latency target (ms). |

---

## Pre-configured providers (35 models, 24 providers)

Use any model without manually entering cost and latency numbers.

```python
from lc_shift import ANTHROPIC, OPENAI, GOOGLE, DEEPSEEK, GROQ, OLLAMA

config = RouterConfig(
    tiers={
        "performance": ANTHROPIC["claude-opus-4-6"],
        "balanced":    OPENAI["gpt-4o"],
        "economy":     GOOGLE["gemini-flash"],
    },
    default_tier="balanced",
    strategy=Strategy.COMPLEXITY,
)
```

### Available providers

| Provider | Keys |
|---|---|
| `ANTHROPIC` | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| `OPENAI` | `gpt-5.5`, `gpt-4o`, `gpt-4o-mini` |
| `GOOGLE` | `gemini-3.1-pro-preview`, `gemini-flash`, `gemini-gemini-cli-flash` |
| `DEEPSEEK` | `deepseek-v4-flash`, `deepseek-v3.2`, `deepseek-r1-huggingface`, `deepseek-v3.2-deepinfra` |
| `MISTRAL` | `mistral-large-latest` |
| `XAI` | `grok-4.3` |
| `MOONSHOT` | `kimi-k2.6`, `kimi-for-coding`, `kimi-k2.5-together` |
| `QWEN` | `qwen3.5-plus` |
| `MINIMAX` | `minimax-m2.7` |
| `NVIDIA` | `nemotron-super-120b` |
| `GROQ` | `llama3-groq` |
| `CEREBRAS` | `zai-glm-4.7` |
| `OPENROUTER` | `auto` |
| `VERCEL` | `claude-opus-via-vercel` |
| `QIANFAN` | `deepseek-v3.2-qianfan` |
| `STEPFUN` | `step-3.5-flash` |
| `XIAOMI` | `mimo-v2-flash` |
| `VOLCENGINE` | `ark-code-latest` |
| `BYTEPLUS` | `ark-code-latest` |
| `GITHUB_COPILOT` | `copilot-default` |
| `OLLAMA` | `llama3.3` *(local, free)* |
| `LMSTUDIO` | `gpt-oss-20b` *(local, free)* |
| `VLLM` | `custom` *(local, free)* |
| `SGLANG` | `custom` *(local, free)* |

### Ready-made presets

```python
from lc_shift import PRESETS

PRESETS["anthropic-3tier"]   # Claude Opus / Sonnet / Haiku
PRESETS["openai-3tier"]      # GPT-5.5 / GPT-4o / GPT-4o-mini
PRESETS["mixed-frontier"]    # Claude Opus / GPT-4o / Gemini Flash
PRESETS["cost-optimized"]    # DeepSeek V3.2 / DeepSeek Flash / Ollama
PRESETS["speed-first"]       # Groq / DeepSeek Flash / vLLM
PRESETS["local-only"]        # Ollama / vLLM / sglang (zero cost)
```

---

## Fallback chains

When a provider goes down, `route_with_fallback()` returns an ordered list of healthy tiers automatically.

```python
from lc_shift import RouterShifter, TierHealth

health = TierHealth(cooldown_seconds=60)
router = RouterShifter(config, health=health)

chain = await router.route_with_fallback(request)

for decision in chain:
    try:
        result = await call_llm(decision.tier)
        router.record_usage(decision.tier_name, input_tokens=200, output_tokens=500)
        break
    except ProviderError as exc:
        router.mark_tier_failed(decision.tier_name)   # skip for 60s
else:
    raise RuntimeError("All tiers exhausted")

print(chain.skipped_tiers)   # ['performance']  — tiers that were degraded
```

Tiers auto-recover after `cooldown_seconds`. You can also call `router.recover_tier("performance")` to clear it manually.

---

## Batch routing

Route multiple prompts concurrently in a single call.

```python
decisions = await router.route_batch([
    ShiftRequest(prompt="Quick question"),
    ShiftRequest(prompt="Deep multi-step analysis with code review..."),
    ShiftRequest(prompt="Translate to French"),
])
# Returns list[RoutingDecision] in the same order, all routed concurrently.
```

---

## Observability hooks

Plug in your own logging, alerting, or OpenTelemetry tracing with zero coupling.

```python
from lc_shift import HookRegistry, RouterShifter

hooks = HookRegistry()

@hooks.on_route
def log_decision(request, decision):
    print(f"[{decision.tier_name}] {decision.overhead_ms:.2f}ms — {decision.reason}")

@hooks.on_usage
async def push_metrics(tier_name, input_tokens, output_tokens):
    await metrics.record(tier_name, input_tokens, output_tokens)

@hooks.on_fallback
async def alert_on_degraded(failed_tier, next_tier, exc):
    await slack.send(f"Provider degraded: {failed_tier} -> {next_tier}")

@hooks.on_error
def log_error(request, exc):
    logger.error(f"Routing failed: {exc}")

router = RouterShifter(config, hooks=hooks)
```

Hooks can be sync or async — both work.

---

## Routing cache

Cache routing decisions for identical prompts to eliminate repeated scoring overhead.

```python
from lc_shift import RoutingCache, RouterShifter

cache = RoutingCache(ttl_seconds=120, max_size=2000)
router = RouterShifter(config, cache=cache)

d1 = await router.route(ShiftRequest(prompt="Explain transformers"))
# overhead: 0.04ms, cache_hit: False

d2 = await router.route(ShiftRequest(prompt="Explain transformers"))
# overhead: 0.00ms, cache_hit: True

print(cache.size)        # 1
print(cache.total_hits)  # 1
```

---

## Cost tracking and metrics

```python
router.record_usage("balanced", input_tokens=500, output_tokens=1200)

snap = router.snapshot()
snap.total_requests          # int
snap.estimated_cost_usd      # float
snap.budget_remaining_usd    # float | None
snap.cache_hit_rate          # float  (0.0 – 1.0)
snap.degraded_tiers          # list[str]
snap.tier_metrics            # dict[str, TierMetrics]

# Per-tier breakdown
m = snap.tier_metrics["balanced"]
m.requests           # int
m.input_tokens       # int
m.output_tokens      # int
m.estimated_cost_usd # float
```

### Budget-aware routing

```python
config = RouterConfig(
    tiers=PRESETS["mixed-frontier"],
    default_tier="balanced",
    strategy=Strategy.COST_AWARE,
    cost_budget_usd=10.00,    # auto-downshift as spend grows
)
```

---

## Custom ModelTier

You can define any provider not in the built-in catalog:

```python
from lc_shift import ModelTier, RouterConfig

config = RouterConfig(
    tiers={
        "my-model": ModelTier(
            name="My Custom Model",
            provider="my-provider",
            model_id="my-model-v1",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.003,
            avg_latency_ms=800,
            max_tokens=8192,
        ),
    },
    default_tier="my-model",
    strategy=Strategy.CASCADE,
)
```

---

## Force a specific tier

```python
decision = await router.route(ShiftRequest(prompt="...", force_tier="performance"))
# reason: "force_tier override"
```

---

## All features together

```python
from lc_shift import (
    RouterShifter, RouterConfig, ShiftRequest, Strategy,
    HookRegistry, RoutingCache, TierHealth,
    PRESETS,
)

hooks = HookRegistry()

@hooks.on_route
def log(request, decision):
    print(f"[{decision.tier_name}] cache={decision.cache_hit}")

router = RouterShifter(
    RouterConfig(tiers=PRESETS["mixed-frontier"], default_tier="balanced"),
    hooks=HookRegistry(),
    cache=RoutingCache(ttl_seconds=60),
    health=TierHealth(cooldown_seconds=30),
)

async with router:
    chain = await router.route_with_fallback(ShiftRequest(prompt="..."))
    decisions = await router.route_batch([ShiftRequest(prompt=p) for p in prompts])
    snap = router.snapshot()
```

---

## Development

```bash
git clone https://github.com/Saimoguloju/lc-shift.git
cd lc-shift
uv sync --dev

uv run pytest -v          # 36 tests, <1s
uv run ruff check src/ tests/
uv run mypy src/
```

CI runs the full matrix: **Ubuntu × macOS × Windows** × **Python 3.11 / 3.12 / 3.13**.

---

## Contributing

Issues and PRs are welcome. If you're adding a new provider preset, edit [`src/lc_shift/providers.py`](src/lc_shift/providers.py) — follow the existing pattern and include approximate cost/latency values.

If you're adding a new routing strategy, subclass `BaseStrategy` in [`src/lc_shift/strategies.py`](src/lc_shift/strategies.py) and register it in `STRATEGY_MAP`.

---

## License

MIT — see [LICENSE](LICENSE).
