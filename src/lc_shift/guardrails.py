"""PII detection & redaction guardrails — pure standard library.

Masks sensitive data (emails, phone numbers, SSNs, credit cards, IP addresses,
cloud/API keys) *before* a prompt is routed or forwarded to an external model, so
secrets never leave your network. Built on stdlib ``re`` only — no models, no
network — keeping lc-shift's zero-dependency promise.

Usage::

    redactor = PIIRedactor()
    result = redactor.redact("email me at jane@acme.com or call 555-123-4567")
    result.text     # 'email me at [REDACTED_EMAIL] or call [REDACTED_PHONE]'
    result.counts   # {'email': 1, 'phone': 1}

Wire it into the proxy with ``serve(..., redactor=PIIRedactor())`` (or
``lc-shift serve --redact-pii``) to scrub every outbound request, or use it as an
``on_route`` style guard in your own pipeline.

Note: regex PII detection is best-effort. Phone/IP patterns can produce false
positives on arbitrary digit strings; disable types you do not need via
``PIIRedactor(types={PIIType.EMAIL, PIIType.SSN})``.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel


class PIIType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    AWS_KEY = "aws_key"
    API_KEY = "api_key"


# Detection order = overlap-resolution priority (earlier wins on overlap).
_PATTERNS: list[tuple[PIIType, re.Pattern[str]]] = [
    (PIIType.AWS_KEY, re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (PIIType.API_KEY, re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (PIIType.EMAIL, re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    (PIIType.SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (PIIType.CREDIT_CARD, re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    (PIIType.IP_ADDRESS, re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    (
        PIIType.PHONE,
        re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    ),
]

DEFAULT_TYPES: frozenset[PIIType] = frozenset(PIIType)


class PIIFinding(BaseModel):
    type: str
    placeholder: str
    start: int
    end: int


class RedactionResult(BaseModel):
    text: str
    findings: list[PIIFinding]
    counts: dict[str, int]

    @property
    def has_pii(self) -> bool:
        return bool(self.findings)


def _luhn_ok(candidate: str) -> bool:
    """Validate a credit-card-like number via the Luhn checksum."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _valid_ipv4(candidate: str) -> bool:
    parts = candidate.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


class PIIRedactor:
    """Detects and redacts PII using a configurable set of pattern types."""

    __slots__ = ("_types", "_placeholder_fmt")

    def __init__(
        self,
        types: set[PIIType] | frozenset[PIIType] | None = None,
        *,
        placeholder_fmt: str = "[REDACTED_{type}]",
    ) -> None:
        self._types = frozenset(types) if types is not None else DEFAULT_TYPES
        self._placeholder_fmt = placeholder_fmt

    def _placeholder(self, pii_type: PIIType) -> str:
        return self._placeholder_fmt.format(type=pii_type.value.upper())

    def detect(self, text: str) -> list[PIIFinding]:
        """Return non-overlapping PII findings, ordered by position."""
        spans: list[PIIFinding] = []
        claimed: list[tuple[int, int]] = []

        for pii_type, pattern in _PATTERNS:
            if pii_type not in self._types:
                continue
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                value = match.group()
                if pii_type is PIIType.CREDIT_CARD and not _luhn_ok(value):
                    continue
                if pii_type is PIIType.IP_ADDRESS and not _valid_ipv4(value):
                    continue
                if any(start < c_end and end > c_start for c_start, c_end in claimed):
                    continue  # overlaps a higher-priority finding
                claimed.append((start, end))
                spans.append(
                    PIIFinding(
                        type=pii_type.value,
                        placeholder=self._placeholder(pii_type),
                        start=start,
                        end=end,
                    )
                )

        spans.sort(key=lambda f: f.start)
        return spans

    def redact(self, text: str) -> RedactionResult:
        """Replace detected PII with type-tagged placeholders."""
        findings = self.detect(text)
        if not findings:
            return RedactionResult(text=text, findings=[], counts={})

        out: list[str] = []
        cursor = 0
        counts: dict[str, int] = {}
        for finding in findings:
            out.append(text[cursor:finding.start])
            out.append(finding.placeholder)
            cursor = finding.end
            counts[finding.type] = counts.get(finding.type, 0) + 1
        out.append(text[cursor:])

        return RedactionResult(text="".join(out), findings=findings, counts=counts)
