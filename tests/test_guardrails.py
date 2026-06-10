from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lc_shift.config import ModelTier, RouterConfig, Strategy
from lc_shift.guardrails import PIIRedactor, PIIType
from lc_shift.mcp import MCPServer
from lc_shift.router import RouterShifter
from lc_shift.server import BackendConfig, create_server, redact_messages


def _single_tier_config() -> RouterConfig:
    return RouterConfig(
        tiers={
            "t": ModelTier(
                name="T", provider="x", model_id="x",
                cost_per_1k_input=0.0, cost_per_1k_output=0.0, avg_latency_ms=1,
            )
        },
        default_tier="t",
    )


class TestDetection:
    def test_email(self) -> None:
        r = PIIRedactor(types={PIIType.EMAIL})
        result = r.redact("contact jane.doe@acme.co.uk please")
        assert result.text == "contact [REDACTED_EMAIL] please"
        assert result.counts == {"email": 1}

    def test_ssn(self) -> None:
        r = PIIRedactor(types={PIIType.SSN})
        assert r.redact("ssn 123-45-6789").text == "ssn [REDACTED_SSN]"

    def test_credit_card_valid_luhn_redacted(self) -> None:
        r = PIIRedactor(types={PIIType.CREDIT_CARD})
        result = r.redact("card 4111111111111111 end")
        assert result.counts == {"credit_card": 1}

    def test_credit_card_invalid_luhn_ignored(self) -> None:
        r = PIIRedactor(types={PIIType.CREDIT_CARD})
        # Fails the Luhn checksum -> not treated as a card.
        assert r.redact("num 1234567890123456 end").has_pii is False

    def test_ip_address_valid(self) -> None:
        r = PIIRedactor(types={PIIType.IP_ADDRESS})
        assert r.redact("host 192.168.1.1").text == "host [REDACTED_IP_ADDRESS]"

    def test_ip_address_rejects_out_of_range_octets(self) -> None:
        r = PIIRedactor(types={PIIType.IP_ADDRESS})
        assert r.redact("ver 999.999.999.999").has_pii is False

    def test_aws_key(self) -> None:
        r = PIIRedactor(types={PIIType.AWS_KEY})
        assert r.redact("key AKIAIOSFODNN7EXAMPLE here").counts == {"aws_key": 1}

    def test_api_key(self) -> None:
        r = PIIRedactor(types={PIIType.API_KEY})
        assert r.redact("token sk-abcdefghij1234567890XYZ done").counts == {"api_key": 1}

    def test_phone(self) -> None:
        r = PIIRedactor(types={PIIType.PHONE})
        assert r.redact("call 555-123-4567 now").text == "call [REDACTED_PHONE] now"


class TestRedactorBehaviour:
    def test_multiple_types_and_counts(self) -> None:
        r = PIIRedactor()
        result = r.redact("mail a@b.com and a@b.com, ssn 123-45-6789")
        assert result.counts == {"email": 2, "ssn": 1}
        assert "a@b.com" not in result.text

    def test_no_pii_returns_unchanged(self) -> None:
        r = PIIRedactor()
        result = r.redact("just a normal sentence")
        assert result.text == "just a normal sentence"
        assert result.has_pii is False

    def test_disabled_type_not_detected(self) -> None:
        r = PIIRedactor(types={PIIType.SSN})  # email disabled
        assert r.redact("mail x@y.com").has_pii is False

    def test_custom_placeholder(self) -> None:
        r = PIIRedactor(types={PIIType.EMAIL}, placeholder_fmt="<{type}>")
        assert r.redact("x@y.com").text == "<EMAIL>"

    def test_findings_have_positions(self) -> None:
        r = PIIRedactor(types={PIIType.EMAIL})
        findings = r.detect("hi x@y.com")
        assert len(findings) == 1
        assert findings[0].type == "email"
        assert findings[0].start == 3


class TestRedactMessages:
    def test_redacts_string_content(self) -> None:
        payload = {"messages": [{"role": "user", "content": "email me jane@acme.com"}]}
        new, counts = redact_messages(payload, PIIRedactor())
        assert counts == {"email": 1}
        assert "jane@acme.com" not in new["messages"][0]["content"]
        assert payload["messages"][0]["content"] == "email me jane@acme.com"  # original untouched

    def test_redacts_multimodal_text_parts(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "ssn 123-45-6789"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ]}
            ]
        }
        new, counts = redact_messages(payload, PIIRedactor())
        assert counts == {"ssn": 1}
        assert new["messages"][0]["content"][0]["text"] == "ssn [REDACTED_SSN]"


class TestMCPRedactTool:
    def test_redact_pii_tool(self) -> None:
        server = MCPServer(RouterShifter(_single_tier_config()))
        resp = server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "redact_pii", "arguments": {"text": "mail a@b.com"}}}
        )
        assert resp is not None
        result = resp["result"]
        assert result["isError"] is False
        assert result["structuredContent"]["counts"] == {"email": 1}
        assert "a@b.com" not in result["structuredContent"]["redacted_text"]

    def test_redact_pii_in_tools_list(self) -> None:
        server = MCPServer(RouterShifter(_single_tier_config()))
        resp = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert resp is not None
        assert "redact_pii" in {t["name"] for t in resp["result"]["tools"]}


# --------------------------------------------------------------------------
# Proxy integration: PII must be scrubbed before reaching the backend
# --------------------------------------------------------------------------
class _EchoBackendHandler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        # Echo the messages we received so the test can assert redaction upstream.
        out = json.dumps({"model": body["model"], "received_messages": body["messages"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture()
def echo_backend() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoBackendHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _make_proxy(three_tier_config: RouterConfig, backend_url: str, *, pii_mode: str) -> tuple[ThreadingHTTPServer, str]:
    router = RouterShifter(three_tier_config.model_copy(update={"strategy": Strategy.COMPLEXITY}))
    server = create_server(
        router,
        BackendConfig(base_url=backend_url),
        host="127.0.0.1",
        port=0,
        redactor=PIIRedactor(),
        pii_mode=pii_mode,  # type: ignore[arg-type]
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class TestProxyRedaction:
    def test_pii_redacted_before_forwarding(
        self, three_tier_config: RouterConfig, echo_backend: str
    ) -> None:
        server, url = _make_proxy(three_tier_config, echo_backend, pii_mode="redact")
        try:
            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=json.dumps(
                    {"model": "auto", "messages": [
                        {"role": "user", "content": "my email is bob@evil.com send help"}
                    ]}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                headers = dict(resp.headers)
            # The backend echo must NOT contain the raw email.
            forwarded = body["received_messages"][0]["content"]
            assert "bob@evil.com" not in forwarded
            assert "[REDACTED_EMAIL]" in forwarded
            assert "email" in headers["x-lc-shift-pii-redacted"]
        finally:
            server.shutdown()
            server.server_close()

    def test_reject_mode_blocks_request(
        self, three_tier_config: RouterConfig, echo_backend: str
    ) -> None:
        server, url = _make_proxy(three_tier_config, echo_backend, pii_mode="reject")
        try:
            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=json.dumps(
                    {"model": "auto", "messages": [{"role": "user", "content": "ssn 123-45-6789"}]}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req, timeout=5)
            assert exc.value.code == 400
        finally:
            server.shutdown()
            server.server_close()
