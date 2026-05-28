# lc-shift

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/mypy-strict-green.svg" alt="Mypy Strict">
  <img src="https://img.shields.io/badge/code%20style-ruff-black.svg" alt="Code Style: Ruff">
  <img src="https://img.shields.io/badge/license-MIT-purple.svg" alt="License">
</p>

`lc-shift` is a provider-agnostic, zero-external-API-dependency LLM routing library. It dynamically directs prompts to the optimal model tier (e.g., Performance, Balanced, Economy) in **under 1ms** on local CPU, maximizing quality while saving up to 80% on API costs.

By running entirely locally without network queries or heavy ML frameworks (like scikit-learn or PyTorch), `lc-shift` introduces negligible overhead to your application middleware.

---

## Key Features

*   🎯 **Intent-based Semantic Routing**: Matches prompt intent against predefined examples using an on-device TF-IDF & Cosine Similarity engine.
*   🧠 **RouteLLM-style Classifier**: Uses a local linear model/logistic regression classifier to compute the probability that a prompt requires a frontier model.
*   ⚡ **Sub-1ms Performance**: Crafted using vanilla Python math utilities to run locally with zero-overhead.
*   📊 **Visual Streamlit Playground**: Includes an interactive dashboard to visualize routing decisions, track simulated costs, and compare strategies.
*   🛡️ **Fallback & Health Cooldowns**: Excludes failing, degraded, or rate-limited endpoints from the pool automatically.
*   🔌 **Observability Hooks**: Register custom lifecycle callbacks (`on_route`, `on_usage`, `on_fallback`, `on_error`) for logging and metrics tracking.

---

## Installation

```bash
pip install lc-shift
```

**Requirements:** Python 3.11+ and Pydantic v2.

---

## Quick Start

### 1. Intent-Based Semantic Routing
Route prompts to specialized tiers based on the semantic intent matching of your reference utterances:

```python
import asyncio
from lc_shift import RouterShifter, RouterConfig, ShiftRequest, Strategy, PRESETS

# Configure routes mapping tiers to example prompt utterances
config = RouterConfig(
    tiers=PRESETS["mixed-frontier"],
    default_tier="balanced",
    strategy=Strategy.SEMANTIC,
    semantic_routes={
        "performance": [
            "Write a recursive algorithm to solve matrix multiplication",
            "Evaluate time and space complexity of quicksort",
            "Prove the correctness of the consensus algorithm"
        ],
        "economy": [
            "Hello!",
            "Tell me a short joke",
            "What is the weather today?",
            "Translate hello to French"
        ]
    }
)

async def main():
    async with RouterShifter(config) as router:
        # Routes to performance tier (Claude Opus)
        d1 = await router.route(ShiftRequest(prompt="Implement compiler optimization in Rust"))
        print(f"Tier: {d1.tier_name} ({d1.reason})")
        
        # Routes to economy tier (Gemini Flash)
        d2 = await router.route(ShiftRequest(prompt="hi, how are you?"))
        print(f"Tier: {d2.tier_name} ({d2.reason})")

asyncio.run(main())
```

### 2. RouteLLM-style Classifier Routing
Score prompts dynamically based on calibrated token weights to route between cheap and expensive models:

```python
import asyncio
from lc_shift import RouterShifter, RouterConfig, ShiftRequest, Strategy, PRESETS

config = RouterConfig(
    tiers=PRESETS["mixed-frontier"],
    default_tier="balanced",
    strategy=Strategy.CLASSIFIER,
    classifier_intercept=-0.5,
    classifier_threshold=0.6,
    classifier_weights={
        "code": 2.5,
        "algorithm": 2.0,
        "mathematics": 1.8,
        "hi": -2.0,
        "hello": -2.0,
        "simple": -1.5
    }
)

async def main():
    async with RouterShifter(config) as router:
        decision = await router.route(ShiftRequest(prompt="Show me a code algorithm"))
        # Routes to performance tier due to high weight sum
        print(f"Tier: {decision.tier_name} ({decision.reason})")

asyncio.run(main())
```

---

## Running the Interactive Playground

To interactively sandbox prompt complexity heuristics, test semantic routes, and visually inspect routing statistics, run the built-in Streamlit dashboard:

```bash
# Sync development dependencies
uv sync

# Launch the playground
uv run streamlit run examples/playground.py
```

---

## Routing Strategies

| Strategy | Goal | Key Parameters |
|---|---|---|
| `SEMANTIC` | Matches prompt intent against example sentences using local cosine similarity. | `semantic_routes` |
| `CLASSIFIER` | Calculates probability score using token weights to route to cheap/expensive models. | `classifier_weights`, `classifier_intercept`, `classifier_threshold` |
| `COMPLEXITY` | Evaluates prompt complexity (length, syntax structure, code blocks) to route simple tasks to cheap tiers. | `complexity_threshold` |
| `COST_AWARE` | Maximizes performance under a set budget, downgrading tiers as consumption approaches limit. | `cost_budget_usd` |
| `LATENCY` | Filters tiers meeting a latency target, routing to the most capable tier within target. | `latency_target_ms` |
| `CASCADE` | Starts with the cheapest tier. Ideal when escalating on output validation failures. | - |

---

## Development & Testing

Setting up the development environment using `uv`:

```bash
git clone https://github.com/Saimoguloju/lc-shift.git
cd lc-shift
uv sync --dev

# Run tests
uv run pytest

# Check code styling & formatting
uv run ruff check src/ tests/ examples/

# Verify strict typing
uv run mypy src/ tests/ examples/
```

---

## License

MIT
