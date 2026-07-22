from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from nanobot.providers.codex_auth import (
    _load_official_codex_token,
    _nanobot_token_path,
    _official_codex_auth_path,
    _refresh_token,
    _write_official_codex_auth,
    format_codex_usage,
    get_codex_usage,
    get_token,
)


def _jwt(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.sig"


def test_load_official_codex_token_from_tokens_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex-home"))
    auth_path = _official_codex_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    access = _jwt(
        {
            "exp": 4_102_444_800,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
        }
    )
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access,
                    "refresh_token": "refresh_123",
                    "account_id": "acct_123",
                },
                "last_refresh": "2026-04-12T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    token = _load_official_codex_token(auth_path)

    assert token is not None
    assert token.access == access
    assert token.refresh == "refresh_123"
    assert token.account_id == "acct_123"
    assert token.expires == 4_102_444_800_000


def test_get_token_imports_official_codex_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    access = _jwt(
        {
            "exp": 4_102_444_800,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_home"},
        }
    )
    auth_path = tmp_path / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access,
                    "refresh_token": "refresh_home",
                    "account_id": "acct_home",
                }
            }
        ),
        encoding="utf-8",
    )

    token = get_token()
    token_path = _nanobot_token_path()

    assert token.access == access
    assert token.refresh == "refresh_home"
    assert token.account_id == "acct_home"
    saved = json.loads(token_path.read_text(encoding="utf-8"))
    assert saved["account_id"] == "acct_home"
    assert saved["refresh"] == "refresh_home"


def test_refresh_token_keeps_existing_refresh_when_endpoint_omits_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200
        text = '{"access_token":"new_access"}'

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "access_token": _jwt(
                    {
                        "exp": 4_102_444_900,
                        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_refresh"},
                    }
                )
            }

    monkeypatch.setattr("nanobot.providers.codex_auth.httpx.post", lambda *args, **kwargs: _Response())

    token = _refresh_token("refresh_existing", "acct_refresh")

    assert token.refresh == "refresh_existing"
    assert token.account_id == "acct_refresh"
    assert token.expires == 4_102_444_900_000


def test_write_official_codex_auth_updates_tokens_block(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"tokens": {"refresh_token": "old"}}), encoding="utf-8")

    _write_official_codex_auth(
        auth_path,
        type("Token", (), {
            "access": "access_123",
            "refresh": "refresh_123",
            "expires": 4_102_444_900_000,
            "account_id": "acct_write",
        })(),
    )

    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    assert payload["tokens"]["access_token"] == "access_123"
    assert payload["tokens"]["refresh_token"] == "refresh_123"
    assert payload["tokens"]["account_id"] == "acct_write"
    assert "last_refresh" in payload


def test_get_codex_usage_uses_oauth_account_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Token:
        access = "access_123"
        account_id = "acct_123"

    class _Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"plan_type": "plus", "rate_limit": {"primary_window": {"used_percent": 25}}}

    captured: dict[str, object] = {}

    def _get(url: str, **kwargs: object) -> _Response:
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr("nanobot.providers.codex_auth.get_token", lambda: _Token())
    monkeypatch.setattr("nanobot.providers.codex_auth.httpx.get", _get)

    payload = get_codex_usage()

    assert payload["plan_type"] == "plus"
    assert captured["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert captured["headers"] == {
        "Authorization": "Bearer access_123",
        "ChatGPT-Account-ID": "acct_123",
        "originator": "codex_cli_rs",
        "Accept": "application/json",
    }


def test_format_codex_usage_renders_windows_and_reset_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.providers.codex_auth.time.time", lambda: 1_700_000_000)

    content = format_codex_usage(
        {
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 25,
                    "limit_window_seconds": 900,
                    "reset_at": 1_700_003_600,
                }
            },
        }
    )

    assert "Codex quota (plus):" in content
    assert "Primary: 75% remaining" in content
    assert "window 15m" not in content
    assert "Primary: 75% remaining\nReset at 2023-11-15 07:13 MYT" in content
