#!/usr/bin/env python3
"""Agentic routing demo — role routing + cheap-first escalation.

    uv run python examples/agent_loop.py

Shows the two agent-native primitives:
  1. Role routing  — each agent step (plan/tool-select/summarize) goes to a tier
     appropriate for its difficulty.
  2. Escalation    — start on the cheapest model and climb only when a validation
     check fails, the canonical cost-saving agent pattern.

No network calls: the "LLM" is a stub so the demo is deterministic. Swap the stub
for a real client (or point an OpenAI SDK at `lc-shift serve`) in production.
"""

from __future__ import annotations

import asyncio

from lc_shift import AgentRouter, ModelTier, RouterConfig, RouterShifter, Strategy, PRESETS


# ---------------------------------------------------------------------------
# 1. Role routing: one tier per agent step
# ---------------------------------------------------------------------------
async def role_routing_demo() -> None:
    print("=" * 60)
    print("1. Role-based routing for an agent loop")
    print("=" * 60)

    config = RouterConfig(
        tiers=PRESETS["mixed-frontier"],
        default_tier="balanced",
        strategy=Strategy.ROLE,
        role_routes={
            "planner": "performance",     # hard reasoning -> frontier model
            "tool_select": "economy",     # cheap, structured -> cheapest model
            "summarize": "economy",
            "reflect": "balanced",
        },
    )
    agent = AgentRouter(RouterShifter(config))

    steps = [
        ("planner", "Break the user goal into an ordered plan of tool calls."),
        ("tool_select", "Pick the next tool: search, calculator, or finish."),
        ("summarize", "Summarize the tool output in one line."),
        ("reflect", "Did we satisfy the user goal? Critique and decide."),
    ]
    for role, prompt in steps:
        decision = await agent.route_step(prompt, role=role)
        print(f"  [{role:<11}] -> {decision.tier_name:<11} ({decision.tier.model_id})")


# ---------------------------------------------------------------------------
# 2. Escalation: cheap-first, climb on validation failure
# ---------------------------------------------------------------------------
async def escalation_demo() -> None:
    print("\n" + "=" * 60)
    print("2. Cheap-first escalation on validation failure")
    print("=" * 60)

    config = RouterConfig(
        tiers=PRESETS["anthropic-3tier"], default_tier="balanced", strategy=Strategy.CASCADE
    )
    agent = AgentRouter(RouterShifter(config))

    # Stub model: only the frontier tier produces valid JSON; weaker tiers "fail".
    def call_model(tier: ModelTier, prompt: str) -> str:
        if tier.name == "Claude Opus 4.6":
            return '{"answer": 42}'
        return "sorry, I'm not sure"

    def is_valid_json(output: str) -> bool:
        import json
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False

    result = await agent.run_with_escalation(
        "Return the answer as JSON.", call_model, validate=is_valid_json
    )

    for i, attempt in enumerate(result.attempts):
        label = "OK " if attempt.accepted else "fail"
        print(f"  attempt {i + 1}: {attempt.tier_name:<11} [{label}] -> {attempt.output!r}")
    print(f"\n  final: tier={result.tier_name}  success={result.success}  escalated={result.escalated}")
    print("  (cheap tiers were tried first; only escalated to the frontier model when needed)")


async def main() -> None:
    await role_routing_demo()
    await escalation_demo()


if __name__ == "__main__":
    asyncio.run(main())
