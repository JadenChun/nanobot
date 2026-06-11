from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from nanobot.providers import copilot_auth


@pytest.fixture(autouse=True)
def _isolated_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    auth_file = tmp_path / "copilot_auth.json"
    monkeypatch.setenv("NANOBOT_COPILOT_AUTH_PATH", str(auth_file))
    return auth_file


def test_is_authenticated_false_when_no_file(_isolated_auth: Path) -> None:
    assert copilot_auth.is_authenticated() is False


def test_get_copilot_bearer_uses_cached_token_when_fresh(_isolated_auth: Path) -> None:
    _isolated_auth.write_text(
        json.dumps(
            {
                "github_token": "gh_long_lived",
                "copilot_token": "tid=cached",
                "copilot_expires_at": int(time.time()) + 3600,
            }
        )
    )
    assert copilot_auth.get_copilot_bearer() == "tid=cached"


def test_get_copilot_bearer_refreshes_when_expired(
    _isolated_auth: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolated_auth.write_text(
        json.dumps(
            {
                "github_token": "gh_long_lived",
                "copilot_token": "tid=stale",
                "copilot_expires_at": int(time.time()) - 60,
            }
        )
    )

    captured: dict[str, object] = {}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "token": "tid=fresh",
                "expires_at": int(time.time()) + 1800,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            },
            request=request,
        )

    monkeypatch.setattr(copilot_auth.httpx, "get", fake_get)

    bearer = copilot_auth.get_copilot_bearer()
    assert bearer == "tid=fresh"
    assert captured["url"] == "https://api.github.com/copilot_internal/v2/token"
    assert captured["headers"]["Authorization"] == "token gh_long_lived"
    # Persisted
    saved = json.loads(_isolated_auth.read_text())
    assert saved["copilot_token"] == "tid=fresh"
    assert saved["copilot_endpoints"] == {"api": "https://api.githubcopilot.com"}


def test_get_copilot_bearer_raises_when_unauthenticated(_isolated_auth: Path) -> None:
    with pytest.raises(RuntimeError, match="not authenticated"):
        copilot_auth.get_copilot_bearer()


def test_get_copilot_bearer_surfaces_entitlement_error(
    _isolated_auth: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolated_auth.write_text(json.dumps({"github_token": "gh_token"}))

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        return httpx.Response(403, text="no copilot", request=httpx.Request("GET", url))

    monkeypatch.setattr(copilot_auth.httpx, "get", fake_get)
    with pytest.raises(RuntimeError, match="not entitled to Copilot"):
        copilot_auth.get_copilot_bearer()


def test_copilot_request_headers_includes_editor_identity() -> None:
    headers = copilot_auth.copilot_request_headers()
    assert headers["Copilot-Integration-Id"] == "vscode-chat"
    assert headers["Editor-Version"].startswith("vscode/")
    assert "Editor-Plugin-Version" in headers
