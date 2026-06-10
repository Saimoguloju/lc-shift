from lc_shift.cache import RoutingCache
from lc_shift.config import ModelTier, RouterConfig, Strategy
from lc_shift.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    LCShiftError,
    RoutingError,
)
from lc_shift.agent import AgentRouter, EscalationAttempt, EscalationResult
from lc_shift.eval import BenchmarkResult, EvalRecord, evaluate, load_dataset
from lc_shift.health import TierHealth
from lc_shift.mcp import MCPServer, serve_stdio
from lc_shift.hooks import HookRegistry
from lc_shift.models import CostSnapshot, FallbackChain, RoutingDecision, ShiftRequest, TierMetrics
from lc_shift.router import RouterShifter
from lc_shift.server import BackendConfig, create_server, serve
from lc_shift import providers
from lc_shift.strategies import (
    ClassifierStrategy,
    EnsembleStrategy,
    KNNStrategy,
    LocalTFIDF,
    RoleStrategy,
    SemanticStrategy,
)
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
    "KNNStrategy",
    "EnsembleStrategy",
    "RoleStrategy",
    # Evaluation harness
    "evaluate",
    "load_dataset",
    "BenchmarkResult",
    "EvalRecord",
    # OpenAI-compatible proxy
    "serve",
    "create_server",
    "BackendConfig",
    # Agentic routing
    "AgentRouter",
    "EscalationResult",
    "EscalationAttempt",
    # MCP server
    "MCPServer",
    "serve_stdio",
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

__version__ = "0.4.0"
