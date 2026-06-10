"""Command-line interface for lc-shift.

Examples::

    lc-shift providers
    lc-shift route "Prove the CAP theorem" --preset mixed-frontier --strategy complexity
    lc-shift bench data.jsonl --preset anthropic-3tier --strategy complexity

Built entirely on the standard library (argparse) to honour the project's
zero-dependency promise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from lc_shift.config import RouterConfig, Strategy
from lc_shift.eval import evaluate, load_dataset
from lc_shift.models import ShiftRequest
from lc_shift.providers import ALL_PROVIDERS, PRESETS
from lc_shift.router import RouterShifter

# Strategies fully configurable from flags alone. Data-driven strategies
# (semantic, classifier, knn, ensemble) need extra config, so they are only
# reachable via ``--config path/to/config.json``.
_FLAG_STRATEGIES = ["complexity", "cost_aware", "cascade", "latency"]


def _pick_default_tier(tier_names: Sequence[str]) -> str:
    for preferred in ("balanced", "economy"):
        if preferred in tier_names:
            return preferred
    return tier_names[0]


def _build_router(args: argparse.Namespace) -> RouterShifter:
    if getattr(args, "config", None):
        config = RouterConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
        return RouterShifter(config)

    if args.preset not in PRESETS:
        raise SystemExit(f"unknown preset '{args.preset}'; choose from {list(PRESETS)}")
    tiers = PRESETS[args.preset]
    config = RouterConfig(
        tiers=tiers,
        default_tier=_pick_default_tier(list(tiers)),
        strategy=Strategy(args.strategy),
        complexity_threshold=args.threshold,
    )
    return RouterShifter(config)


def _cmd_providers(_: argparse.Namespace) -> int:
    print(f"{len(ALL_PROVIDERS)} providers:\n")
    for key, tier in ALL_PROVIDERS.items():
        tag = "  [LOCAL/FREE]" if tier.cost_per_1k_input == 0.0 else ""
        print(f"  {key:<48} ${tier.cost_per_1k_input:.5f}/${tier.cost_per_1k_output:.5f}{tag}")
    print(f"\nPresets: {list(PRESETS)}")
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    router = _build_router(args)

    async def _run() -> None:
        decision = await router.route(ShiftRequest(prompt=args.prompt))
        print(f"Tier:   {decision.tier_name} ({decision.tier.provider}/{decision.tier.model_id})")
        print(f"Reason: {decision.reason}")
        print(f"Cost:   ${decision.tier.cost_per_1k_input:.5f}/1k in, "
              f"${decision.tier.cost_per_1k_output:.5f}/1k out")
        print(f"Time:   {decision.overhead_ms:.4f} ms")

    asyncio.run(_run())
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    router = _build_router(args)
    dataset = load_dataset(args.dataset)

    async def _run() -> int:
        result = await evaluate(router, dataset)
        print(result.format_report())
        return 0

    return asyncio.run(_run())


def _cmd_serve(args: argparse.Namespace) -> int:
    from lc_shift.guardrails import PIIRedactor
    from lc_shift.server import BackendConfig, serve

    router = _build_router(args)
    backend = BackendConfig(base_url=args.backend, api_key=args.api_key)
    redactor = PIIRedactor() if args.redact_pii else None
    serve(router, backend, host=args.host, port=args.port, redactor=redactor, pii_mode=args.pii_mode)
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    # stdout is the JSON-RPC channel here, so never print to it.
    from lc_shift.mcp import serve_stdio

    router = _build_router(args)
    print(
        f"lc-shift MCP server on stdio | strategy={router.config.strategy.value} "
        f"| tiers={list(router.config.tiers)}",
        file=sys.stderr,
    )
    serve_stdio(router)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lc-shift", description="Local, zero-dependency LLM router.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--preset", default="mixed-frontier", help="tier preset (see `lc-shift providers`)")
    common.add_argument(
        "--strategy",
        default="complexity",
        choices=_FLAG_STRATEGIES,
        help="routing strategy (data-driven strategies require --config)",
    )
    common.add_argument("--threshold", type=float, default=0.5, help="complexity threshold (0..1)")
    common.add_argument(
        "--config",
        default=None,
        help="path to a RouterConfig JSON file (overrides --preset/--strategy)",
    )

    p_prov = sub.add_parser("providers", help="list all built-in providers and presets")
    p_prov.set_defaults(func=_cmd_providers)

    p_route = sub.add_parser("route", parents=[common], help="route a single prompt")
    p_route.add_argument("prompt", help="the prompt to route")
    p_route.set_defaults(func=_cmd_route)

    p_bench = sub.add_parser("bench", parents=[common], help="benchmark a strategy over a JSONL dataset")
    p_bench.add_argument("dataset", help="path to a JSONL file of {prompt, ideal_tier} records")
    p_bench.set_defaults(func=_cmd_bench)

    p_serve = sub.add_parser(
        "serve",
        parents=[common],
        help="run a drop-in OpenAI-compatible routing proxy",
    )
    p_serve.add_argument(
        "--backend",
        required=True,
        help="OpenAI-compatible backend base URL to forward to, "
        "e.g. http://localhost:11434/v1 (Ollama) or https://api.openai.com/v1",
    )
    p_serve.add_argument("--api-key", default=None, help="API key forwarded to the backend")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="bind port (default 8000)")
    p_serve.add_argument(
        "--redact-pii",
        action="store_true",
        help="scrub PII (emails, phones, SSNs, cards, keys) from requests before forwarding",
    )
    p_serve.add_argument(
        "--pii-mode",
        default="redact",
        choices=["redact", "reject"],
        help="redact PII in place (default) or reject requests containing PII with HTTP 400",
    )
    p_serve.set_defaults(func=_cmd_serve)

    p_mcp = sub.add_parser(
        "mcp",
        parents=[common],
        help="run a Model Context Protocol (MCP) server over stdio",
    )
    p_mcp.set_defaults(func=_cmd_mcp)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func  # set by every subparser
    result: int = func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
