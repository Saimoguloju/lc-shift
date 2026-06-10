"""Drop-in OpenAI-compatible routing proxy — pure standard library.

Point any OpenAI SDK at this server and it transparently picks the optimal model
tier with lc-shift's sub-1ms local router, rewrites the ``model`` field, and
forwards the request to a configured OpenAI-compatible backend (OpenAI, Ollama,
vLLM, LiteLLM, OpenRouter, …). No application code changes required::

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-...")
    client.chat.completions.create(model="auto", messages=[...])

Built entirely on ``http.server`` + ``urllib`` so the proxy keeps lc-shift's
zero-runtime-dependency promise. The response carries ``x-lc-shift-tier`` and
``x-lc-shift-model`` headers so callers can see what was chosen.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import BaseModel, Field

from lc_shift.models import RoutingDecision, ShiftRequest
from lc_shift.router import RouterShifter

_CHAT_PATH = "/chat/completions"


class BackendConfig(BaseModel):
    """Where the proxy forwards requests after routing."""

    base_url: str = Field(min_length=1)  # e.g. "http://localhost:11434/v1"
    api_key: str | None = None
    timeout: float = Field(default=60.0, gt=0)

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + _CHAT_PATH


def extract_prompt(payload: dict[str, Any]) -> str:
    """Pull routable text from an OpenAI chat-completions request body.

    Uses the most recent user message; ``content`` may be a plain string or the
    multimodal list form, in which case text parts are concatenated.
    """
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                joined = " ".join(t for t in parts if t)
                if joined:
                    return joined
    # Fall back to any string content so we always have something to route on.
    for message in messages:
        if isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


async def select_decision(router: RouterShifter, payload: dict[str, Any]) -> RoutingDecision:
    """Route the request body and return the chosen tier decision."""
    prompt = extract_prompt(payload)
    if not prompt:
        raise ValueError("no routable text found in 'messages'")
    return await router.route(ShiftRequest(prompt=prompt))


def rewrite_payload(payload: dict[str, Any], decision: RoutingDecision) -> dict[str, Any]:
    """Return a copy of the request body with ``model`` set to the routed model."""
    rewritten = dict(payload)
    rewritten["model"] = decision.tier.model_id
    return rewritten


class _RouterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        router: RouterShifter,
        backend: BackendConfig,
    ) -> None:
        super().__init__(address, handler)
        self.router = router
        self.backend = backend


class _ProxyHandler(BaseHTTPRequestHandler):
    server: _RouterHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *_: Any) -> None:  # silence default stderr spam
        pass

    # -- helpers ----------------------------------------------------------
    def _send_json(self, status: int, body: dict[str, Any], extra: dict[str, str] | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok"})
            return
        if self.path.rstrip("/") == "/v1/models":
            tiers = self.server.router.config.tiers
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": t.model_id, "object": "model", "owned_by": t.provider}
                        for t in tiers.values()
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if not self.path.endswith(_CHAT_PATH):
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}})
            return

        try:
            decision = asyncio.run(select_decision(self.server.router, payload))
        except ValueError as exc:
            self._send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        except Exception as exc:  # routing should never 500 silently
            self._send_json(500, {"error": {"message": f"routing failed: {exc}", "type": "lc_shift_error"}})
            return

        rewritten = rewrite_payload(payload, decision)
        self._forward(rewritten, decision)

    # -- upstream forwarding ---------------------------------------------
    def _forward(self, payload: dict[str, Any], decision: RoutingDecision) -> None:
        backend = self.server.backend
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        elif backend.api_key:
            headers["Authorization"] = f"Bearer {backend.api_key}"

        request = urllib.request.Request(
            backend.chat_url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        lc_headers = {
            "x-lc-shift-tier": decision.tier_name,
            "x-lc-shift-model": decision.tier.model_id,
            "x-lc-shift-overhead-ms": f"{decision.overhead_ms:.4f}",
        }

        try:
            with urllib.request.urlopen(request, timeout=backend.timeout) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                for key, value in lc_headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as exc:  # forward upstream error verbatim
            body = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            for key, value in lc_headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except urllib.error.URLError as exc:
            self._send_json(
                502,
                {"error": {"message": f"backend unreachable: {exc.reason}", "type": "upstream_error"}},
                extra=lc_headers,
            )


def create_server(
    router: RouterShifter,
    backend: BackendConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> _RouterHTTPServer:
    """Create (but do not start) the proxy HTTP server."""
    return _RouterHTTPServer((host, port), _ProxyHandler, router, backend)


def serve(
    router: RouterShifter,
    backend: BackendConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the proxy and serve forever (blocking)."""
    httpd = create_server(router, backend, host=host, port=port)
    print(f"lc-shift proxy on http://{host}:{port}  ->  {backend.base_url}")
    print(f"  strategy: {router.config.strategy.value} | tiers: {list(router.config.tiers)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
