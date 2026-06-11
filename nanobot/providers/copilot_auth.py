"""GitHub Copilot OAuth (device flow) and bearer-token management.

Two-stage credential model used by GitHub Copilot (and matched by every
open-source Copilot integration — copilot.vim, aider, opencode, etc.):

1. **GitHub OAuth token** (long-lived; obtained via GitHub device-code flow
   against the public Copilot client ID). Stored in ``~/.nanobot/copilot_auth.json``.
2. **Copilot internal bearer** (short-lived, ~30 minutes; minted by exchanging
   the GitHub token at ``https://api.github.com/copilot_internal/v2/token``).
   Cached in memory and on disk; refreshed automatically a few minutes before
   expiry.

Public surface:

* :func:`device_login` — run the interactive device-code flow and persist the
  GitHub token.
* :func:`get_copilot_bearer` — return a currently-valid Copilot bearer,
  refreshing on demand. Used by the OpenAI-compatible provider as its
  ``token_provider`` callback.
* :func:`copilot_request_headers` — extra HTTP headers Copilot requires.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

# Public GitHub Copilot OAuth client ID. Identical value is published in
# copilot.vim, opencode, aider, neovim CopilotChat, etc. Not a secret.
_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
_DEVICE_CODE_URL = "https://github.com/login/device/code"
_OAUTH_TOKEN_URL = "https://github.com/login/oauth/access_token"
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
_DEVICE_SCOPE = "read:user"

# Identify ourselves as a VS Code-style Copilot client. The Copilot API
# rejects requests that don't look like a known editor integration.
_EDITOR_VERSION = "vscode/1.95.0"
_EDITOR_PLUGIN_VERSION = "copilot-chat/0.22.0"
_COPILOT_INTEGRATION_ID = "vscode-chat"
_USER_AGENT = "GitHubCopilotChat/0.22.0"

# Refresh the short-lived Copilot bearer a few minutes before it actually
# expires so in-flight requests don't race the expiry boundary.
_REFRESH_EARLY_SECONDS = 300

_LOCK = threading.Lock()


def _auth_path() -> Path:
    override = os.environ.get("NANOBOT_COPILOT_AUTH_PATH")
    if override:
        return Path(override).expanduser()
    home = Path(os.environ.get("HOME", str(Path.home())))
    return home / ".nanobot" / "copilot_auth.json"


def _load_auth() -> dict[str, Any]:
    path = _auth_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_auth(data: dict[str, Any]) -> None:
    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Device-code flow
# ---------------------------------------------------------------------------


def device_login(
    *,
    on_user_code: Callable[[str, str], None] | None = None,
    poll_timeout: float = 600.0,
) -> dict[str, Any]:
    """Run the GitHub device-code flow for the Copilot client and persist.

    Args:
        on_user_code: Optional callback invoked with ``(user_code, verification_uri)``
            so the caller can display the prompt. If omitted, prints to stdout.
        poll_timeout: Max seconds to wait for the user to approve.

    Returns:
        The full saved auth dict.
    """
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            _DEVICE_CODE_URL,
            data={"client_id": _COPILOT_CLIENT_ID, "scope": _DEVICE_SCOPE},
            headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"GitHub device code request failed: {resp.status_code} {resp.text}"
            )
        device = resp.json()
        device_code = device.get("device_code")
        user_code = device.get("user_code")
        verification_uri = device.get("verification_uri") or "https://github.com/login/device"
        interval = max(1, int(device.get("interval") or 5))
        if not device_code or not user_code:
            raise RuntimeError(f"Malformed device-code response: {device}")

        if on_user_code is not None:
            on_user_code(user_code, verification_uri)
        else:
            print(f"\nGo to {verification_uri} and enter code: {user_code}\n")

        deadline = time.time() + poll_timeout
        access_token: str | None = None
        while time.time() < deadline:
            time.sleep(interval)
            poll = client.post(
                _OAUTH_TOKEN_URL,
                data={
                    "client_id": _COPILOT_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
            )
            if poll.status_code != 200:
                raise RuntimeError(
                    f"GitHub token poll failed: {poll.status_code} {poll.text}"
                )
            data = poll.json()
            err = data.get("error")
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval += 5
                continue
            if err in {"expired_token", "access_denied"}:
                raise RuntimeError(f"GitHub OAuth flow ended: {err}")
            if err:
                raise RuntimeError(f"GitHub OAuth error: {err} ({data.get('error_description')})")
            access_token = data.get("access_token")
            if access_token:
                break

        if not access_token:
            raise RuntimeError("Timed out waiting for GitHub device-code approval")

    saved = {"github_token": access_token}
    _save_auth(saved)
    # Eagerly mint a Copilot bearer so we surface entitlement errors at login
    # time (e.g. the GitHub account has no Copilot subscription).
    _refresh_copilot_bearer(saved)
    return _load_auth()


# ---------------------------------------------------------------------------
# Copilot bearer
# ---------------------------------------------------------------------------


def _refresh_copilot_bearer(auth: dict[str, Any]) -> dict[str, Any]:
    """Exchange the GitHub OAuth token for a fresh Copilot bearer; persist."""
    github_token = auth.get("github_token")
    if not github_token:
        raise RuntimeError(
            "GitHub Copilot is not authenticated. Run `nanobot provider login github-copilot`."
        )

    resp = httpx.get(
        _COPILOT_TOKEN_URL,
        headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
            "Editor-Version": _EDITOR_VERSION,
            "Editor-Plugin-Version": _EDITOR_PLUGIN_VERSION,
        },
        timeout=30.0,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "GitHub token is no longer valid. Re-run `nanobot provider login github-copilot`."
        )
    if resp.status_code == 403:
        raise RuntimeError(
            "GitHub account is not entitled to Copilot. "
            "Ensure the account has an active Copilot subscription."
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Copilot token exchange failed: {resp.status_code} {resp.text}"
        )

    payload = resp.json()
    bearer = payload.get("token")
    expires_at = payload.get("expires_at")
    if not bearer or not isinstance(expires_at, int):
        raise RuntimeError(f"Malformed Copilot token response: {payload}")

    auth["copilot_token"] = bearer
    auth["copilot_expires_at"] = expires_at
    auth["copilot_endpoints"] = payload.get("endpoints") or {}
    _save_auth(auth)
    return auth


def get_copilot_bearer(min_ttl_seconds: int = _REFRESH_EARLY_SECONDS) -> str:
    """Return a valid Copilot bearer token, refreshing if near expiry.

    Safe to call from multiple threads / coroutines — refresh is serialized.
    """
    with _LOCK:
        auth = _load_auth()
        token = auth.get("copilot_token")
        expires_at = auth.get("copilot_expires_at") or 0
        if token and expires_at - int(time.time()) > min_ttl_seconds:
            return token
        auth = _refresh_copilot_bearer(auth)
        return auth["copilot_token"]


def copilot_request_headers() -> dict[str, str]:
    """HTTP headers Copilot expects on every request."""
    return {
        "Editor-Version": _EDITOR_VERSION,
        "Editor-Plugin-Version": _EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": _COPILOT_INTEGRATION_ID,
        "User-Agent": _USER_AGENT,
    }


def is_authenticated() -> bool:
    return bool(_load_auth().get("github_token"))
