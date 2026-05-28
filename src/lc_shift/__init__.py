from lc_shift.cache import RoutingCache
from lc_shift.config import ModelTier, RouterConfig, Strategy
from lc_shift.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    LCShiftError,
    RoutingError,
)
from lc_shift.health import TierHealth
from lc_shift.hooks import HookRegistry
from lc_shift.models import CostSnapshot, FallbackChain, RoutingDecision, ShiftRequest, TierMetrics
from lc_shift.router import RouterShifter
from lc_shift import providers
from lc_shift.strategies import LocalTFIDF, SemanticStrategy, ClassifierStrategy
from lc_shift.providers import (
    ALL_PROVIDERS,
    PRESETS,
    ANTHROPIC,
    OPENAI,
    GOOGLE,
    DEEPSEEK,
    MISTRAL,
    XAI,
    MOONSHOT,
    QWEN,
    MINIMAX,
    NVIDIA,
    GROQ,
    CEREBRAS,
    OPENROUTER,
    VERCEL,
    QIANFAN,
    STEPFUN,
    XIAOMI,
    VOLCENGINE,
    BYTEPLUS,
    GITHUB_COPILOT,
    OLLAMA,
    LMSTUDIO,
    VLLM,
    SGLANG,
)

__all__ = [
    # Core
    "RouterShifter",
    "RouterConfig",
    "ModelTier",
    "Strategy",
    "ShiftRequest",
    "RoutingDecision",
    "FallbackChain",
    "CostSnapshot",
    "TierMetrics",
    # Exceptions
    "LCShiftError",
    "ConfigurationError",
    "RoutingError",
    "BudgetExhaustedError",
    # New feature classes
    "HookRegistry",
    "RoutingCache",
    "TierHealth",
    "LocalTFIDF",
    "SemanticStrategy",
    "ClassifierStrategy",
    # Providers module

    "providers",
    "ALL_PROVIDERS",
    "PRESETS",
    # Provider dicts
    "ANTHROPIC",
    "OPENAI",
    "GOOGLE",
    "DEEPSEEK",
    "MISTRAL",
    "XAI",
    "MOONSHOT",
    "QWEN",
    "MINIMAX",
    "NVIDIA",
    "GROQ",
    "CEREBRAS",
    "OPENROUTER",
    "VERCEL",
    "QIANFAN",
    "STEPFUN",
    "XIAOMI",
    "VOLCENGINE",
    "BYTEPLUS",
    "GITHUB_COPILOT",
    "OLLAMA",
    "LMSTUDIO",
    "VLLM",
    "SGLANG",
]

__version__ = "0.1.1"
