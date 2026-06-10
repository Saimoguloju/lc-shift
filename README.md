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

*   🔌 **Drop-in OpenAI-Compatible Proxy**: Point any OpenAI SDK at lc-shift and it transparently routes each request to the optimal model tier — **no code changes**. Pure standard library (`http.server` + `urllib`), so the proxy keeps the zero-dependency promise.
*   🎯 **Intent-based Semantic Routing**: Matches prompt intent against predefined examples using an on-device TF-IDF & Cosine Similarity engine.
*   🧠 **RouteLLM-style Classifier**: Uses a local linear model/logistic regression classifier to compute the probability that a prompt requires a frontier model.
*   🤝 **kNN Router with Online Learning**: Non-parametric k-nearest-neighbour routing that adapts at runtime via `router.learn()`. Recent research ([*"When Simple kNN Beats Complex Learned Routers"*, arXiv:2505.12601](https://arxiv.org/pdf/2505.12601)) shows this approach rivals heavyweight learned routers.
*   🗳️ **Ensemble Routing**: Combine complexity, classifier, and semantic signals into a single weighted vote to hedge any one signal's blind spots.
*   📏 **Built-in Evaluation Harness**: Score any strategy on a labelled dataset with RouteLLM-style **cost-quality metrics** (accuracy, quality-preserved, cost-savings %) — the evidence real router benchmarks (RouteLLM, RouterBench) are judged on.
*   🖥️ **Zero-dependency CLI**: `lc-shift route`, `lc-shift bench`, and `lc-shift providers` for routing and benchmarking from the terminal.
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

## Drop-in OpenAI-Compatible Proxy

The fastest way to use lc-shift: run it as a proxy and point your **existing** OpenAI
code at it. Every request is routed locally (sub-1ms) to the best tier, the `model`
field is rewritten, and the call is forwarded to your backend (OpenAI, Ollama, vLLM,
LiteLLM, OpenRouter, …). No application code changes.

```bash
# Route in front of Ollama (local), using the complexity strategy
lc-shift serve --backend http://localhost:11434/v1 --preset cost-optimized --strategy complexity

# …or in front of OpenAI
lc-shift serve --backend https://api.openai.com/v1 --api-key $OPENAI_API_KEY --preset openai-3tier
```

```python
from openai import OpenAI

# The ONLY change: base_url points at lc-shift instead of the provider.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-...")

resp = client.chat.completions.create(
    model="auto",                                  # lc-shift overrides this per-prompt
    messages=[{"role": "user", "content": "What is 2 + 2?"}],   # -> cheap tier
)
print(resp.model)  # the model lc-shift actually routed to
```

Every response includes `x-lc-shift-tier` and `x-lc-shift-model` headers so you can see
the decision. Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.

> Use `--config path/to/config.json` instead of `--preset/--strategy` to drive the proxy
> with the `knn`, `semantic`, `classifier`, or `ensemble` strategies.

### Run with Docker

```bash
docker build -t lc-shift .
docker run -p 8000:8000 lc-shift \
  serve --backend http://host.docker.internal:11434/v1 --host 0.0.0.0
```

---

## Quick Start (Library)

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
| `KNN` | k-nearest-neighbour voting over labelled examples in local TF-IDF space; supports online learning. | `knn_examples`, `knn_k` |
| `ENSEMBLE` | Weighted vote across complexity, classifier, and semantic signals. | `ensemble_weights` |
| `COMPLEXITY` | Evaluates prompt complexity (length, syntax structure, code blocks) to route simple tasks to cheap tiers. | `complexity_threshold` |
| `COST_AWARE` | Maximizes performance under a set budget, downgrading tiers as consumption approaches limit. | `cost_budget_usd` |
| `LATENCY` | Filters tiers meeting a latency target, routing to the most capable tier within target. | `latency_target_ms` |
| `CASCADE` | Starts with the cheapest tier. Ideal when escalating on output validation failures. | - |

---

## Adaptive kNN Routing & Online Learning

The kNN router stores labelled example prompts per tier and routes by similarity-weighted
voting over the `k` nearest neighbours. It learns from feedback at runtime — no retraining,
no network, no ML framework:

```python
import asyncio
from lc_shift import RouterShifter, RouterConfig, ShiftRequest, Strategy, PRESETS

config = RouterConfig(
    tiers=PRESETS["mixed-frontier"],
    default_tier="balanced",
    strategy=Strategy.KNN,
    knn_k=3,
    knn_examples={
        "performance": ["prove the correctness of a consensus algorithm",
                        "analyze and derive the complexity of quicksort"],
        "economy":     ["hello there", "tell me a joke", "capital of japan"],
    },
)

async def main():
    async with RouterShifter(config) as router:
        d = await router.route(ShiftRequest(prompt="what is the weather like?"))
        print(d.tier_name)                       # -> default/economy

        # Teach the router from feedback; future similar prompts follow.
        router.learn("what is the weather like?", "economy")
        d = await router.route(ShiftRequest(prompt="what is the weather today?"))
        print(d.tier_name)                       # -> economy

asyncio.run(main())
```

---

## Benchmarking Strategies (Cost-Quality Evaluation)

Routers should be judged on the **cost-quality trade-off**, not features. The built-in
evaluation harness scores any strategy on a labelled JSONL dataset and reports how much
cost it saves versus always calling the strongest tier, and how much quality it preserves.

```bash
uv run python examples/benchmark.py
```

Sample run (20 prompts, `mixed-frontier` preset):

| strategy | accuracy | quality preserved | cost savings | overhead |
|---|---|---|---|---|
| complexity | 45.0% | 45.0% | 94.6% | 0.011 ms |
| semantic | 80.0% | **100.0%** | 58.7% | 0.031 ms |
| classifier | 70.0% | 70.0% | 69.7% | 0.009 ms |
| **knn** | **80.0%** | **100.0%** | 58.7% | 0.032 ms |
| ensemble | 65.0% | 70.0% | 64.7% | 0.065 ms |

> *quality preserved* = share of prompts never under-routed (no quality loss).
> *cost savings* = reduction vs. sending every prompt to the strongest tier.

Programmatically:

```python
from lc_shift import evaluate, load_dataset, RouterShifter, RouterConfig, Strategy, PRESETS

dataset = load_dataset("examples/benchmark_dataset.jsonl")
router = RouterShifter(RouterConfig(tiers=PRESETS["mixed-frontier"],
                                    default_tier="balanced", strategy=Strategy.COMPLEXITY))
result = await evaluate(router, dataset)
print(result.format_report())
```

---

## Command-Line Interface

```bash
# List all 35 providers and presets
lc-shift providers

# Route a single prompt
lc-shift route "Prove the CAP theorem" --preset mixed-frontier --strategy complexity

# Benchmark a strategy on a dataset
lc-shift bench examples/benchmark_dataset.jsonl --preset mixed-frontier --strategy cascade

# Use a full RouterConfig JSON (required for knn / semantic / classifier / ensemble)
lc-shift bench examples/benchmark_dataset.jsonl --config examples/knn_config.json
```

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
