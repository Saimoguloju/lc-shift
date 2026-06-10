from __future__ import annotations

import io
import json

import pytest

from lc_shift.config import RouterConfig, Strategy
from lc_shift.mcp import PROTOCOL_VERSION, MCPServer
from lc_shift.router import RouterShifter


@pytest.fixture()
def server(three_tier_config: RouterConfig) -> MCPServer:
    router = RouterShifter(
        three_tier_config.model_copy(
            update={"strategy": Strategy.COMPLEXITY, "complexity_threshold": 0.3}
        )
    )
    return MCPServer(router)


def _req(method: str, msg_id: int | None = 1, **params: object) -> dict[str, object]:
    msg: dict[str, object] = {"jsonrpc": "2.0", "method": method}
    if msg_id is not None:
        msg["id"] = msg_id
    if params:
        msg["params"] = params
    return msg


class TestLifecycle:
    def test_initialize_echoes_protocol_version(self, server: MCPServer) -> None:
        resp = server.handle_message(_req("initialize", protocolVersion="2025-06-18"))
        assert resp is not None
        result = resp["result"]
        assert result["protocolVersion"] == "2025-06-18"
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "lc-shift"

    def test_initialize_defaults_version_when_absent(self, server: MCPServer) -> None:
        resp = server.handle_message(_req("initialize"))
        assert resp is not None
        assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION

    def test_initialized_notification_no_response(self, server: MCPServer) -> None:
        assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_ping(self, server: MCPServer) -> None:
        resp = server.handle_message(_req("ping"))
        assert resp is not None and resp["result"] == {}

    def test_unknown_method_errors(self, server: MCPServer) -> None:
        resp = server.handle_message(_req("does/not/exist"))
        assert resp is not None
        assert resp["error"]["code"] == -32601


class TestTools:
    def test_tools_list(self, server: MCPServer) -> None:
        resp = server.handle_message(_req("tools/list"))
        assert resp is not None
        names = {t["name"] for t in resp["result"]["tools"]}
        assert names == {"route_prompt", "estimate_cost", "list_tiers", "redact_pii"}
        for tool in resp["result"]["tools"]:
            assert "inputSchema" in tool and tool["inputSchema"]["type"] == "object"

    def test_route_prompt_simple(self, server: MCPServer) -> None:
        resp = server.handle_message(
            _req("tools/call", name="route_prompt", arguments={"prompt": "hi there"})
        )
        assert resp is not None
        result = resp["result"]
        assert result["isError"] is False
        assert result["structuredContent"]["tier"] == "economy"
        assert result["content"][0]["type"] == "text"

    def test_route_prompt_complex(self, server: MCPServer) -> None:
        hard = (
            "Analyze and compare the trade-off and explain why. "
            "```python\ndef f(): pass\n```\n1. First 2. Second 3. Third"
        )
        resp = server.handle_message(
            _req("tools/call", name="route_prompt", arguments={"prompt": hard})
        )
        assert resp is not None
        assert resp["result"]["structuredContent"]["tier"] == "performance"

    def test_route_prompt_missing_arg_is_tool_error(self, server: MCPServer) -> None:
        resp = server.handle_message(_req("tools/call", name="route_prompt", arguments={}))
        assert resp is not None
        # Tool execution errors are in-band (isError), not protocol errors.
        assert resp["result"]["isError"] is True

    def test_estimate_cost_from_prompt(self, server: MCPServer) -> None:
        resp = server.handle_message(
            _req("tools/call", name="estimate_cost", arguments={"prompt": "hello", "output_tokens": 100})
        )
        assert resp is not None
        costs = resp["result"]["structuredContent"]["cost_by_tier_usd"]
        assert set(costs) == {"performance", "balanced", "economy"}
        assert costs["performance"] > costs["economy"]

    def test_estimate_cost_with_explicit_tokens(self, server: MCPServer) -> None:
        resp = server.handle_message(
            _req("tools/call", name="estimate_cost", arguments={"input_tokens": 1000, "output_tokens": 1000})
        )
        assert resp is not None
        assert resp["result"]["structuredContent"]["input_tokens"] == 1000

    def test_list_tiers(self, server: MCPServer) -> None:
        resp = server.handle_message(_req("tools/call", name="list_tiers", arguments={}))
        assert resp is not None
        tiers = resp["result"]["structuredContent"]["tiers"]
        assert "performance" in tiers and tiers["performance"]["provider"] == "anthropic"

    def test_unknown_tool_is_protocol_error(self, server: MCPServer) -> None:
        resp = server.handle_message(_req("tools/call", name="ghost", arguments={}))
        assert resp is not None
        assert resp["error"]["code"] == -32602


class TestTransport:
    def test_serve_reads_lines_and_writes_responses(self, server: MCPServer) -> None:
        stdin = io.StringIO(
            json.dumps(_req("initialize", protocolVersion="2025-06-18")) + "\n"
            + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
            + json.dumps(_req("tools/call", msg_id=2, name="route_prompt", arguments={"prompt": "hi"})) + "\n"
        )
        stdout = io.StringIO()
        server.serve(stdin=stdin, stdout=stdout)

        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        # initialize -> response, notification -> no response, tools/call -> response
        assert len(lines) == 2
        assert lines[0]["id"] == 1
        assert lines[1]["id"] == 2
        assert lines[1]["result"]["structuredContent"]["tier"] == "economy"

    def test_serve_handles_invalid_json(self, server: MCPServer) -> None:
        stdout = io.StringIO()
        server.serve(stdin=io.StringIO("{not valid json\n"), stdout=stdout)
        resp = json.loads(stdout.getvalue())
        assert resp["error"]["code"] == -32700
