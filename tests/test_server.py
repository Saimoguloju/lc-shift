from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lc_shift.config import RouterConfig, Strategy
from lc_shift.models import RoutingDecision
from lc_shift.router import RouterShifter
from lc_shift.server import (
    BackendConfig,
    create_server,
    extract_prompt,
    rewrite_payload,
    select_decision,
)


class TestPromptExtraction:
    def test_string_content(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello world"}]}
        assert extract_prompt(payload) == "hello world"

    def test_uses_last_user_message(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
        }
        assert extract_prompt(payload) == "second"

    def test_multimodal_text_parts(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                        {"type": "text", "text": "this"},
                    ],
                }
            ]
        }
        assert extract_prompt(payload) == "describe this"

    def test_empty_when_no_user(self) -> None:
        assert extract_prompt({"messages": [{"role": "system", "content": "x"}]}) == "x"
        assert extract_prompt({"messages": []}) == ""


class TestRewrite:
    def test_rewrites_model_to_routed_model_id(self, three_tier_config: RouterConfig) -> None:
        tier = three_tier_config.tiers["performance"]
        decision = RoutingDecision(
            tier_name="performance", tier=tier, reason="x", overhead_ms=0.1
        )
        out = rewrite_payload({"model": "auto", "messages": []}, decision)
        assert out["model"] == tier.model_id
        assert out["model"] != "auto"

    @pytest.mark.asyncio
    async def test_select_decision_routes(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(
            three_tier_config.model_copy(
                update={"strategy": Strategy.COMPLEXITY, "complexity_threshold": 0.3}
            )
        )
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        decision = await select_decision(router, payload)
        assert decision.tier_name == "economy"

    @pytest.mark.asyncio
    async def test_select_decision_empty_raises(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(three_tier_config)
        with pytest.raises(ValueError, match="no routable text"):
            await select_decision(router, {"messages": []})


# --------------------------------------------------------------------------
# Integration: real proxy in front of a fake OpenAI-compatible backend
# --------------------------------------------------------------------------
class _FakeBackendHandler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        # Echo back the model we received so the test can assert the rewrite.
        out = json.dumps(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "_seen_auth": self.headers.get("Authorization"),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture()
def fake_backend() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def proxy(three_tier_config: RouterConfig, fake_backend: str) -> Iterator[str]:
    router = RouterShifter(
        three_tier_config.model_copy(
            update={"strategy": Strategy.COMPLEXITY, "complexity_threshold": 0.3}
        )
    )
    backend = BackendConfig(base_url=fake_backend, api_key="sk-test")
    server = create_server(router, backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _post(url: str, payload: dict[str, object]) -> tuple[int, dict[str, object], dict[str, str]]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read()), dict(resp.headers)


class TestProxyIntegration:
    def test_simple_prompt_routes_to_economy_model(self, proxy: str) -> None:
        status, body, headers = _post(
            f"{proxy}/v1/chat/completions",
            {"model": "auto", "messages": [{"role": "user", "content": "hi there"}]},
        )
        assert status == 200
        # Backend echoes the model the proxy forwarded -> the economy model id.
        assert body["model"] == "claude-haiku-4-5"
        assert headers["x-lc-shift-tier"] == "economy"
        assert headers["x-lc-shift-model"] == "claude-haiku-4-5"

    def test_complex_prompt_routes_to_performance_model(self, proxy: str) -> None:
        hard = (
            "Analyze and compare the trade-off and explain why. "
            "```python\ndef f(): pass\n```\n1. First 2. Second 3. Third"
        )
        _, body, headers = _post(
            f"{proxy}/v1/chat/completions",
            {"model": "auto", "messages": [{"role": "user", "content": hard}]},
        )
        assert headers["x-lc-shift-tier"] == "performance"
        assert body["model"] == "claude-opus-4-6"

    def test_api_key_forwarded_when_client_omits_it(self, proxy: str) -> None:
        _, body, _ = _post(
            f"{proxy}/v1/chat/completions",
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert body["_seen_auth"] == "Bearer sk-test"

    def test_health_endpoint(self, proxy: str) -> None:
        with urllib.request.urlopen(f"{proxy}/health", timeout=5) as resp:
            assert json.loads(resp.read())["status"] == "ok"

    def test_models_endpoint(self, proxy: str) -> None:
        with urllib.request.urlopen(f"{proxy}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
        ids = {m["id"] for m in data["data"]}
        assert "claude-opus-4-6" in ids

    def test_invalid_json_returns_400(self, proxy: str) -> None:
        req = urllib.request.Request(
            f"{proxy}/v1/chat/completions", data=b"{not json", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 400
