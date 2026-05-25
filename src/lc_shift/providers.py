"""
Pre-configured ModelTier presets for all OpenClaw-supported providers.

Usage:
    from lc_shift.providers import ANTHROPIC, OPENAI, GOOGLE, PRESETS
    from lc_shift import RouterConfig, Strategy

    config = RouterConfig(
        tiers={
            "performance": ANTHROPIC["claude-opus-4-6"],
            "balanced":    OPENAI["gpt-4o"],
            "economy":     GOOGLE["gemini-flash"],
        },
        default_tier="balanced",
        strategy=Strategy.COMPLEXITY,
    )

Costs are per 1 000 tokens (input / output).
Latencies are approximate averages in milliseconds.
Local providers (Ollama, LM Studio, vLLM, sglang) have zero cost.
"""

from __future__ import annotations

from lc_shift.config import ModelTier

# ---------------------------------------------------------------------------
# Anthropic — Claude family
# ---------------------------------------------------------------------------
ANTHROPIC: dict[str, ModelTier] = {
    "claude-opus-4-6": ModelTier(
        name="Claude Opus 4.6",
        provider="anthropic",
        model_id="claude-opus-4-6",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        avg_latency_ms=2500,
        max_tokens=8192,
    ),
    "claude-sonnet-4-6": ModelTier(
        name="Claude Sonnet 4.6",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        avg_latency_ms=1200,
        max_tokens=8192,
    ),
    "claude-haiku-4-5": ModelTier(
        name="Claude Haiku 4.5",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
        avg_latency_ms=400,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# OpenAI — GPT family
# ---------------------------------------------------------------------------
OPENAI: dict[str, ModelTier] = {
    "gpt-5.5": ModelTier(
        name="GPT-5.5",
        provider="openai",
        model_id="gpt-5.5",
        cost_per_1k_input=0.010,
        cost_per_1k_output=0.030,
        avg_latency_ms=2000,
        max_tokens=16384,
    ),
    "gpt-4o": ModelTier(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.010,
        avg_latency_ms=1000,
        max_tokens=8192,
    ),
    "gpt-4o-mini": ModelTier(
        name="GPT-4o Mini",
        provider="openai",
        model_id="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        avg_latency_ms=300,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# Google — Gemini family
# ---------------------------------------------------------------------------
GOOGLE: dict[str, ModelTier] = {
    "gemini-3.1-pro-preview": ModelTier(
        name="Gemini 3.1 Pro Preview",
        provider="google",
        model_id="gemini-3.1-pro-preview",
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
        avg_latency_ms=1500,
        max_tokens=8192,
    ),
    "gemini-flash": ModelTier(
        name="Gemini Flash",
        provider="google",
        model_id="gemini-3-flash-preview",
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
        avg_latency_ms=500,
        max_tokens=8192,
    ),
    "gemini-gemini-cli-flash": ModelTier(
        name="Gemini CLI Flash",
        provider="google-gemini-cli",
        model_id="google-gemini-cli/gemini-3-flash-preview",
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
        avg_latency_ms=500,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------
DEEPSEEK: dict[str, ModelTier] = {
    "deepseek-v4-flash": ModelTier(
        name="DeepSeek V4 Flash",
        provider="deepseek",
        model_id="deepseek-v4-flash",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        avg_latency_ms=600,
        max_tokens=8192,
    ),
    "deepseek-v3.2": ModelTier(
        name="DeepSeek V3.2",
        provider="deepseek",
        model_id="deepseek-v3.2",
        cost_per_1k_input=0.00027,
        cost_per_1k_output=0.0011,
        avg_latency_ms=1200,
        max_tokens=8192,
    ),
    "deepseek-r1-huggingface": ModelTier(
        name="DeepSeek R1 (HuggingFace)",
        provider="huggingface",
        model_id="deepseek-ai/DeepSeek-R1",
        cost_per_1k_input=0.0004,
        cost_per_1k_output=0.0016,
        avg_latency_ms=2000,
        max_tokens=8192,
    ),
    "deepseek-v3.2-deepinfra": ModelTier(
        name="DeepSeek V3.2 (DeepInfra)",
        provider="deepinfra",
        model_id="deepseek-ai/DeepSeek-V3.2",
        cost_per_1k_input=0.00027,
        cost_per_1k_output=0.0011,
        avg_latency_ms=1100,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------
MISTRAL: dict[str, ModelTier] = {
    "mistral-large-latest": ModelTier(
        name="Mistral Large",
        provider="mistral",
        model_id="mistral-large-latest",
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
        avg_latency_ms=1500,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# xAI — Grok family
# ---------------------------------------------------------------------------
XAI: dict[str, ModelTier] = {
    "grok-4.3": ModelTier(
        name="Grok 4.3",
        provider="xai",
        model_id="grok-4.3",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        avg_latency_ms=1800,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# Moonshot / Kimi
# ---------------------------------------------------------------------------
MOONSHOT: dict[str, ModelTier] = {
    "kimi-k2.6": ModelTier(
        name="Kimi K2.6",
        provider="moonshot",
        model_id="kimi-k2.6",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003,
        avg_latency_ms=1000,
        max_tokens=8192,
    ),
    "kimi-for-coding": ModelTier(
        name="Kimi for Coding",
        provider="kimi",
        model_id="kimi-for-coding",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003,
        avg_latency_ms=1000,
        max_tokens=8192,
    ),
    "kimi-k2.5-together": ModelTier(
        name="Kimi K2.5 (Together AI)",
        provider="together",
        model_id="moonshotai/Kimi-K2.5",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003,
        avg_latency_ms=1200,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# Alibaba — Qwen family
# ---------------------------------------------------------------------------
QWEN: dict[str, ModelTier] = {
    "qwen3.5-plus": ModelTier(
        name="Qwen 3.5 Plus",
        provider="qwen",
        model_id="qwen3.5-plus",
        cost_per_1k_input=0.0004,
        cost_per_1k_output=0.0012,
        avg_latency_ms=800,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# MiniMax
# ---------------------------------------------------------------------------
MINIMAX: dict[str, ModelTier] = {
    "minimax-m2.7": ModelTier(
        name="MiniMax M2.7",
        provider="minimax",
        model_id="MiniMax-M2.7",
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.002,
        avg_latency_ms=900,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------
NVIDIA: dict[str, ModelTier] = {
    "nemotron-super-120b": ModelTier(
        name="NVIDIA Nemotron Super 120B",
        provider="nvidia",
        model_id="nvidia/nemotron-3-super-120b-a12b",
        cost_per_1k_input=0.004,
        cost_per_1k_output=0.012,
        avg_latency_ms=2000,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# Groq — ultra-fast inference
# ---------------------------------------------------------------------------
GROQ: dict[str, ModelTier] = {
    "llama3-groq": ModelTier(
        name="LLaMA 3.1 70B (Groq)",
        provider="groq",
        model_id="llama-3.1-70b-versatile",
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
        avg_latency_ms=200,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# Cerebras — fast inference
# ---------------------------------------------------------------------------
CEREBRAS: dict[str, ModelTier] = {
    "zai-glm-4.7": ModelTier(
        name="ZAI GLM 4.7 (Cerebras)",
        provider="cerebras",
        model_id="zai-glm-4.7",
        cost_per_1k_input=0.0006,
        cost_per_1k_output=0.0016,
        avg_latency_ms=250,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# OpenRouter — multi-model aggregator
# ---------------------------------------------------------------------------
OPENROUTER: dict[str, ModelTier] = {
    "auto": ModelTier(
        name="OpenRouter Auto",
        provider="openrouter",
        model_id="openrouter/auto",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003,
        avg_latency_ms=1500,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# Vercel AI Gateway — proxy aggregator
# ---------------------------------------------------------------------------
VERCEL: dict[str, ModelTier] = {
    "claude-opus-via-vercel": ModelTier(
        name="Claude Opus via Vercel AI Gateway",
        provider="vercel-ai-gateway",
        model_id="vercel-ai-gateway/anthropic/claude-opus-4.6",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        avg_latency_ms=2700,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# Baidu — Qianfan
# ---------------------------------------------------------------------------
QIANFAN: dict[str, ModelTier] = {
    "deepseek-v3.2-qianfan": ModelTier(
        name="DeepSeek V3.2 (Qianfan/Baidu)",
        provider="qianfan",
        model_id="deepseek-v3.2",
        cost_per_1k_input=0.00027,
        cost_per_1k_output=0.0011,
        avg_latency_ms=1300,
        max_tokens=8192,
    ),
}

# ---------------------------------------------------------------------------
# Stepfun
# ---------------------------------------------------------------------------
STEPFUN: dict[str, ModelTier] = {
    "step-3.5-flash": ModelTier(
        name="Step 3.5 Flash",
        provider="stepfun",
        model_id="step-3.5-flash",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        avg_latency_ms=700,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# Xiaomi
# ---------------------------------------------------------------------------
XIAOMI: dict[str, ModelTier] = {
    "mimo-v2-flash": ModelTier(
        name="MiMo V2 Flash",
        provider="xiaomi",
        model_id="mimo-v2-flash",
        cost_per_1k_input=0.0003,
        cost_per_1k_output=0.0009,
        avg_latency_ms=450,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# Volcengine / ByteDance — Doubao / Ark
# ---------------------------------------------------------------------------
VOLCENGINE: dict[str, ModelTier] = {
    "ark-code-latest": ModelTier(
        name="Ark Code Latest (Volcengine)",
        provider="volcengine",
        model_id="ark-code-latest",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        avg_latency_ms=800,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# BytePlus — international Ark access
# ---------------------------------------------------------------------------
BYTEPLUS: dict[str, ModelTier] = {
    "ark-code-latest": ModelTier(
        name="Ark Code Latest (BytePlus)",
        provider="byteplus",
        model_id="ark-code-latest",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        avg_latency_ms=850,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# GitHub Copilot
# ---------------------------------------------------------------------------
GITHUB_COPILOT: dict[str, ModelTier] = {
    "copilot-default": ModelTier(
        name="GitHub Copilot",
        provider="github-copilot",
        model_id="gpt-4o",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=1000,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# Local / Self-Hosted — zero cost
# ---------------------------------------------------------------------------
OLLAMA: dict[str, ModelTier] = {
    "llama3.3": ModelTier(
        name="LLaMA 3.3 (Ollama)",
        provider="ollama",
        model_id="llama3.3",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=500,
        max_tokens=4096,
    ),
}

LMSTUDIO: dict[str, ModelTier] = {
    "gpt-oss-20b": ModelTier(
        name="GPT OSS 20B (LM Studio)",
        provider="lmstudio",
        model_id="openai/gpt-oss-20b",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=500,
        max_tokens=4096,
    ),
}

VLLM: dict[str, ModelTier] = {
    "custom": ModelTier(
        name="Custom Model (vLLM)",
        provider="vllm",
        model_id="your-model-id",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=200,
        max_tokens=4096,
    ),
}

SGLANG: dict[str, ModelTier] = {
    "custom": ModelTier(
        name="Custom Model (sglang)",
        provider="sglang",
        model_id="your-model-id",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=180,
        max_tokens=4096,
    ),
}

# ---------------------------------------------------------------------------
# PRESETS — ready-made tier combinations for common use cases
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, ModelTier]] = {
    # All-Anthropic Claude tiers
    "anthropic-3tier": {
        "performance": ANTHROPIC["claude-opus-4-6"],
        "balanced": ANTHROPIC["claude-sonnet-4-6"],
        "economy": ANTHROPIC["claude-haiku-4-5"],
    },
    # All-OpenAI GPT tiers
    "openai-3tier": {
        "performance": OPENAI["gpt-5.5"],
        "balanced": OPENAI["gpt-4o"],
        "economy": OPENAI["gpt-4o-mini"],
    },
    # Mixed frontier providers
    "mixed-frontier": {
        "performance": ANTHROPIC["claude-opus-4-6"],
        "balanced": OPENAI["gpt-4o"],
        "economy": GOOGLE["gemini-flash"],
    },
    # Cost-optimized with local fallback
    "cost-optimized": {
        "performance": DEEPSEEK["deepseek-v3.2"],
        "balanced": DEEPSEEK["deepseek-v4-flash"],
        "economy": OLLAMA["llama3.3"],
    },
    # Speed-first (ultra-low latency)
    "speed-first": {
        "performance": GROQ["llama3-groq"],
        "balanced": DEEPSEEK["deepseek-v4-flash"],
        "economy": VLLM["custom"],
    },
    # Fully local / air-gapped
    "local-only": {
        "primary": OLLAMA["llama3.3"],
        "fast": VLLM["custom"],
        "ultra-fast": SGLANG["custom"],
    },
}

# ---------------------------------------------------------------------------
# ALL_PROVIDERS — flat lookup of every ModelTier by "provider/model-key"
# ---------------------------------------------------------------------------
ALL_PROVIDERS: dict[str, ModelTier] = {
    **{f"anthropic/{k}": v for k, v in ANTHROPIC.items()},
    **{f"openai/{k}": v for k, v in OPENAI.items()},
    **{f"google/{k}": v for k, v in GOOGLE.items()},
    **{f"deepseek/{k}": v for k, v in DEEPSEEK.items()},
    **{f"mistral/{k}": v for k, v in MISTRAL.items()},
    **{f"xai/{k}": v for k, v in XAI.items()},
    **{f"moonshot/{k}": v for k, v in MOONSHOT.items()},
    **{f"qwen/{k}": v for k, v in QWEN.items()},
    **{f"minimax/{k}": v for k, v in MINIMAX.items()},
    **{f"nvidia/{k}": v for k, v in NVIDIA.items()},
    **{f"groq/{k}": v for k, v in GROQ.items()},
    **{f"cerebras/{k}": v for k, v in CEREBRAS.items()},
    **{f"openrouter/{k}": v for k, v in OPENROUTER.items()},
    **{f"vercel/{k}": v for k, v in VERCEL.items()},
    **{f"qianfan/{k}": v for k, v in QIANFAN.items()},
    **{f"stepfun/{k}": v for k, v in STEPFUN.items()},
    **{f"xiaomi/{k}": v for k, v in XIAOMI.items()},
    **{f"volcengine/{k}": v for k, v in VOLCENGINE.items()},
    **{f"byteplus/{k}": v for k, v in BYTEPLUS.items()},
    **{f"github-copilot/{k}": v for k, v in GITHUB_COPILOT.items()},
    **{f"ollama/{k}": v for k, v in OLLAMA.items()},
    **{f"lmstudio/{k}": v for k, v in LMSTUDIO.items()},
    **{f"vllm/{k}": v for k, v in VLLM.items()},
    **{f"sglang/{k}": v for k, v in SGLANG.items()},
}
