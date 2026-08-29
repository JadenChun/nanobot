"""Lightweight credential detector for the git auto-sync safety layer.

This module is a small, self-contained version of the analytics-side secret
detector. It exists in the Nanobot runtime so the context auto-sync can
refuse to push commits that contain credential-bearing content, without
introducing a runtime dependency on the context repository itself.

The detector only emits REDACTED fingerprints. The full credential value
is never returned, logged, or written to error messages.
"""

from __future__ import annotations

import re
from typing import Any


SECRET_QUERY_PARAMS = frozenset({"access_token", "appsecret_proof", "client_secret"})


# Plain-text patterns that always indicate a credential. Each match is fully
# redacted. Order matters: longer / more specific patterns first.
PLAINTEXT_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(Authorization:\s*Bearer\s+)([A-Za-z0-9._\-]+)", re.IGNORECASE
    ),
    re.compile(r"(access_token=)([A-Za-z0-9._\-]+)", re.IGNORECASE),
    re.compile(r"(appsecret_proof=)([A-Za-z0-9._\-]+)", re.IGNORECASE),
    re.compile(r"(client_secret=)([A-Za-z0-9._\-]+)", re.IGNORECASE),
    # Facebook page / user access tokens: EAA prefix followed by 60+ chars of
    # base64-ish content. The 60-char minimum keeps the false-positive rate
    # essentially zero (real tokens are 150+ chars).
    re.compile(r"\bEAA[A-Za-z0-9]{60,}\b"),
)


def _fingerprint(value: str, *, prefix: str) -> str:
    """Return a redacted fingerprint of a secret value. Never includes the value."""
    if not value:
        return f"{prefix}***"
    if len(value) <= 4:
        return f"{prefix}{'*' * len(value)}"
    return f"{prefix}{value[:4]}...{value[-4:]}"


def find_secrets_in_blob(blob: str) -> list[dict[str, str]]:
    """Find probable credentials in a single text blob (commit file content).

    Returns a list of hits. Each hit has keys: ``kind``, ``path`` (the
    matched substring's prefix label), ``field``, ``fingerprint``. The full
    credential value is never included.
    """
    if not isinstance(blob, str) or not blob:
        return []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for param in SECRET_QUERY_PARAMS:
        m = re.search(
            rf"({re.escape(param)}=)([A-Za-z0-9._\-]+)", blob, re.IGNORECASE
        )
        if m:
            value = m.group(2)
            key = f"{param}={value}"
            if key not in seen:
                seen.add(key)
                hits.append(
                    {
                        "kind": "url_query_param",
                        "path": "blob",
                        "field": param,
                        "fingerprint": _fingerprint(value, prefix=param + "="),
                    }
                )
    m = re.search(
        r"(Authorization:\s*Bearer\s+)([A-Za-z0-9._\-]+)", blob, re.IGNORECASE
    )
    if m:
        value = m.group(2)
        key = f"Bearer:{value}"
        if key not in seen:
            seen.add(key)
            hits.append(
                {
                    "kind": "plaintext_pattern",
                    "path": "blob",
                    "field": "Authorization",
                    "fingerprint": _fingerprint(value, prefix="Bearer "),
                }
            )
    m = re.search(r"\b(EAA[A-Za-z0-9]{60,})\b", blob)
    if m:
        value = m.group(1)
        key = f"EAA:{value}"
        if key not in seen:
            seen.add(key)
            hits.append(
                {
                    "kind": "facebook_eaa_token",
                    "path": "blob",
                    "field": "value",
                    "fingerprint": _fingerprint(value, prefix="EAA"),
                }
            )
    return hits


def find_secrets(value: Any) -> list[dict[str, str]]:
    """Find probable credentials anywhere in a payload (recursive)."""
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _emit(hit: dict[str, str], key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        hits.append(hit)

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                child_path = f"{path}.{k}" if path else k
                if (
                    isinstance(k, str)
                    and k.lower() == "authorization"
                    and isinstance(v, str)
                ):
                    m = re.search(
                        r"(?:Bearer\s+)([A-Za-z0-9._\-]+)", v, re.IGNORECASE
                    )
                    if m:
                        _emit(
                            {
                                "kind": "plaintext_pattern",
                                "path": child_path,
                                "field": "Authorization",
                                "fingerprint": _fingerprint(
                                    m.group(1), prefix="Bearer "
                                ),
                            },
                            m.group(1),
                        )
                        continue
                _walk(v, child_path)
            return
        if isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]")
            return
        if isinstance(node, str):
            for h in find_secrets_in_blob(node):
                _emit(h, f"{path}:{h.get('fingerprint', '')}")
            return

    _walk(value, "")
    return hits


__all__ = ["find_secrets", "find_secrets_in_blob", "SECRET_QUERY_PARAMS"]
