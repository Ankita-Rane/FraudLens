"""Deterministic, auditable input controls for the guarded research condition.

These rules are deliberately narrow. They identify common instruction-override and
secret-exfiltration requests; they are not claimed to be a complete security product.
Configurations A and B record the same signals for evaluation, while only C enforces
blocking so the experimental conditions remain distinct.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class SafetyFinding:
    code: str
    category: str
    severity: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bignore\s+(all\s+)?(previous|prior|system|developer)\s+instructions?\b", "instruction_override"),
    (r"\b(reveal|print|show|return)\s+(the\s+)?(system|developer)\s+prompt\b", "prompt_exfiltration"),
    (r"\b(disregard|bypass|disable)\s+(the\s+)?(guardrails?|policy|safety|rules?)\b", "guardrail_bypass"),
    (r"\b(output|reveal|show)\s+(the\s+)?(hidden|evaluation|ground[- ]truth)\s+labels?\b", "label_exfiltration"),
    (r"\b(api[-_ ]?keys?|passwords?|access[-_ ]?tokens?|private keys?)\b", "secret_request"),
    (r"\b(identify|name|reveal)\s+(the\s+)?(real\s+)?cardholder\b", "cardholder_identity_request"),
    (r"\b(full\s+card\s+number|cvv|social\s+security\s+number|ssn)\b", "sensitive_financial_identity_request"),
)

_SCOPE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(guarantee|prove|confirm)\s+(that\s+)?(this|the transaction)\s+is\s+fraud", "fraud_certainty_request"),
    (r"\b(give|provide)\s+(me\s+)?legal advice\b", "legal_advice_request"),
    (r"\b(binding|our bank'?s|institution[- ]specific)\s+(rule|policy|procedure)s?\b", "institution_policy_request"),
    (r"\b(automatically|immediately)\s+(decline|block|close|freeze)\b", "automated_adverse_action"),
    (r"\bautomatic\s+adverse\s+action\b", "automated_adverse_action"),
)


def assess_question(question: str) -> list[SafetyFinding]:
    """Return deterministic input findings without inferring user intent."""
    normalized = " ".join(question.split()).lower()
    findings: list[SafetyFinding] = []
    for pattern, code in _INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            findings.append(SafetyFinding(
                code=code,
                category="prompt_security",
                severity="block",
                description="The question contains an instruction-override or protected-data request.",
            ))
    for pattern, code in _SCOPE_PATTERNS:
        if re.search(pattern, normalized):
            findings.append(SafetyFinding(
                code=code,
                category="policy_scope",
                severity="caution",
                description=(
                    "The request exceeds what public research sources and a fraud-risk "
                    "score can establish; a scoped answer and human review are required."
                ),
            ))
    unique: dict[str, SafetyFinding] = {finding.code: finding for finding in findings}
    return list(unique.values())


def blocking_findings(findings: Iterable[SafetyFinding]) -> list[SafetyFinding]:
    return [finding for finding in findings if finding.severity == "block"]


def findings_payload(findings: Iterable[SafetyFinding]) -> list[dict[str, Any]]:
    return [finding.to_dict() for finding in findings]
