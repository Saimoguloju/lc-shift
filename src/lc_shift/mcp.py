"""Native Model Context Protocol (MCP) server — pure standard library.

Exposes lc-shift's routing brain as MCP tools so any MCP host (Claude Desktop,
Cursor, an agent runtime) can ask *"which model should this prompt go to, and
what will it cost?"* during a session.

This is a from-scratch implementation of MCP's JSON-RPC 2.0 / stdio transport
(spec revision ``2025-06-18``) — no `mcp` SDK, no dependencies — so the server
honours lc-shift's zero-dependency promise. Run it with ``lc-shift mcp``.

Tools exposed:
    * ``route_prompt``   — route a prompt and return the chosen tier + model + cost
    * ``estimate_cost``  — cost of a prompt across every configured tier
    * ``list_tiers``     — the configured tiers and their pricing
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import IO, Any

from lc_shift.guardrails import PIIRedactor
from lc_shift.models import ShiftRequest
from lc_shift.router import RouterShifter
from lc_shift.strategies import estimate_token_count

PROTOCOL_VERSION = "2025-06-18"
_PARSE_ERROR = -32700
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602


class MCPServer:
    """A minimal, spec-compliant MCP server over newline-delimited JSON-RPC."""

    __slots__ = ("_router", "_name", "_version", "_redactor")

    def __init__(self, router: RouterShifter, *, name: str = "lc-shift", version: str = "0.5.0") -> None:
        self._router = router
        self._name = name
        self._version = version
        self._redactor = PIIRedactor()

    # -- transport --------------------------------------------------------
    def serve(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        """Read JSON-RPC messages line-by-line from stdin until EOF."""
        rx = stdin or sys.stdin
        tx = stdout or sys.stdout
        for line in rx:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._write(tx, self._error(None, _PARSE_ERROR, "parse error"))
                continue
            response = self.handle_message(message)
            if response is not None:
                self._write(tx, response)

    @staticmethod
    def _write(tx: IO[str], payload: dict[str, Any]) -> None:
        tx.write(json.dumps(payload) + "\n")
        tx.flush()

    # -- dispatch ---------------------------------------------------------
    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC message; returns a response, or None for notifications."""
        method = message.get("method")
        msg_id = message.get("id")

        # Notifications have no id and never get a response.
        if msg_id is None and isinstance(method, str) and method.startswith("notifications/"):
            return None

        if method == "initialize":
            return self._ok(msg_id, self._initialize(message.get("params") or {}))
        if method == "ping":
            return self._ok(msg_id, {})
        if method == "tools/list":
            return self._ok(msg_id, {"tools": self._tool_definitions()})
        if method == "tools/call":
            return self._tools_call(msg_id, message.get("params") or {})
        if msg_id is None:
            return None  # unknown notification — ignore
        return self._error(msg_id, _METHOD_NOT_FOUND, f"method not found: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if isinstance(requested, str) else PROTOCOL_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self._name, "version": self._version},
            "instructions": (
                "lc-shift routing tools. Use route_prompt to choose the optimal model "
                "tier for a prompt, estimate_cost to compare per-tier cost, and "
                "list_tiers to inspect the configured models."
            ),
        }

    # -- tools ------------------------------------------------------------
    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "route_prompt",
                "title": "Route a prompt to the optimal model tier",
                "description": (
                    "Run lc-shift's local router on a prompt and return the chosen tier, "
                    "model id, provider, the reason, and the per-1k-token cost."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string", "description": "The prompt to route"}},
                    "required": ["prompt"],
                },
            },
            {
                "name": "estimate_cost",
                "title": "Estimate prompt cost across all tiers",
                "description": "Estimate the USD cost of a prompt on every configured tier.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Prompt to size (used if input_tokens omitted)"},
                        "input_tokens": {"type": "integer", "description": "Override input token count"},
                        "output_tokens": {"type": "integer", "description": "Expected output tokens (default 500)"},
                    },
                },
            },
            {
                "name": "list_tiers",
                "title": "List configured tiers",
                "description": "Return the configured tiers with provider, model id, and pricing.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "redact_pii",
                "title": "Redact PII from text",
                "description": (
                    "Detect and mask PII (emails, phones, SSNs, credit cards, IPs, API keys) "
                    "in text before it is sent to a model. Returns the redacted text and counts."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Text to redact"}},
                    "required": ["text"],
                },
            },
        ]

    def _tools_call(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "route_prompt":
                text, structured = self._tool_route_prompt(args)
            elif name == "estimate_cost":
                text, structured = self._tool_estimate_cost(args)
            elif name == "list_tiers":
                text, structured = self._tool_list_tiers()
            elif name == "redact_pii":
                text, structured = self._tool_redact_pii(args)
            else:
                return self._error(msg_id, _INVALID_PARAMS, f"unknown tool: {name}")
        except ValueError as exc:
            # Tool execution error — reported in-band per the MCP spec.
            return self._ok(
                msg_id,
                {"content": [{"type": "text", "text": str(exc)}], "isError": True},
            )
        return self._ok(
            msg_id,
            {
                "content": [{"type": "text", "text": text}],
                "structuredContent": structured,
                "isError": False,
            },
        )

    def _tool_route_prompt(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("'prompt' is required and must be a non-empty string")
        decision = asyncio.run(self._router.route(ShiftRequest(prompt=prompt)))
        tier = decision.tier
        structured = {
            "tier": decision.tier_name,
            "model_id": tier.model_id,
            "provider": tier.provider,
            "reason": decision.reason,
            "cost_per_1k_input": tier.cost_per_1k_input,
            "cost_per_1k_output": tier.cost_per_1k_output,
            "overhead_ms": round(decision.overhead_ms, 4),
        }
        text = (
            f"Routed to '{decision.tier_name}' -> {tier.provider}/{tier.model_id}\n"
            f"Reason: {decision.reason}\n"
            f"Cost: ${tier.cost_per_1k_input:.5f}/1k in, ${tier.cost_per_1k_output:.5f}/1k out"
        )
        return text, structured

    def _tool_estimate_cost(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        input_tokens = args.get("input_tokens")
        if input_tokens is None:
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("provide either 'input_tokens' or a non-empty 'prompt'")
            input_tokens = estimate_token_count(prompt)
        output_tokens = args.get("output_tokens", 500)

        rows: dict[str, float] = {}
        for name, tier in self._router.config.tiers.items():
            cost = (input_tokens / 1000) * tier.cost_per_1k_input + (
                output_tokens / 1000
            ) * tier.cost_per_1k_output
            rows[name] = round(cost, 6)

        structured = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_by_tier_usd": rows,
        }
        lines = [f"Estimated cost for {input_tokens} in / {output_tokens} out tokens:"]
        lines += [f"  {name:<14} ${cost:.6f}" for name, cost in rows.items()]
        return "\n".join(lines), structured

    def _tool_list_tiers(self) -> tuple[str, dict[str, Any]]:
        tiers = self._router.config.tiers
        structured = {
            "default_tier": self._router.config.default_tier,
            "strategy": self._router.config.strategy.value,
            "tiers": {
                name: {
                    "provider": t.provider,
                    "model_id": t.model_id,
                    "cost_per_1k_input": t.cost_per_1k_input,
                    "cost_per_1k_output": t.cost_per_1k_output,
                    "avg_latency_ms": t.avg_latency_ms,
                }
                for name, t in tiers.items()
            },
        }
        lines = [f"{len(tiers)} tiers (strategy: {self._router.config.strategy.value}):"]
        lines += [f"  {name:<14} {t.provider}/{t.model_id}" for name, t in tiers.items()]
        return "\n".join(lines), structured

    def _tool_redact_pii(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text = args.get("text")
        if not isinstance(text, str):
            raise ValueError("'text' is required and must be a string")
        result = self._redactor.redact(text)
        structured = {
            "redacted_text": result.text,
            "counts": result.counts,
            "has_pii": result.has_pii,
        }
        return result.text, structured

    # -- JSON-RPC envelopes ----------------------------------------------
    @staticmethod
    def _ok(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def serve_stdio(router: RouterShifter) -> None:
    """Start the MCP server on stdio (blocking)."""
    MCPServer(router).serve()
