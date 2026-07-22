"""Compatibility helpers for OpenAI Codex OAuth tokens."""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from oauth_cli_kit.models import OAuthToken
from oauth_cli_kit.providers import OPENAI_CODEX_PROVIDER

_REFRESH_EARLY_SECONDS = 300
DEFAULT_CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"


class CodexUsageError(RuntimeError):
    """Raised when the Codex OAuth usage endpoint cannot be read."""


def get_token(min_ttl_seconds: int = 60) -> OAuthToken:
    """Load a usable Codex token from Nanobot or official Codex auth storage."""
    token_path = _nanobot_token_path()
    token = _load_nanobot_token(token_path)
    now_ms = _now_ms()
    if token and token.expires - now_ms > min_ttl_seconds * 1000:
        return token

    auth_path = _official_codex_auth_path()
    official = _load_official_codex_token(auth_path)
    if official:
        _save_nanobot_token(token_path, official)
        if official.expires - now_ms > min_ttl_seconds * 1000:
            return official
        token = official

    if token and token.refresh:
        refreshed = _refresh_token(token.refresh, token.account_id)
        _save_nanobot_token(token_path, refreshed)
        if auth_path.exists():
            _write_official_codex_auth(auth_path, refreshed)
        return refreshed

    if token:
        raise RuntimeError(
            "Codex access token is expired and no refresh token is available. "
            "Run `codex login` on a machine with a browser and copy its auth file to this host."
        )

    raise RuntimeError(
        "OAuth credentials not found. Run `codex login` locally and copy "
        "`~/.codex/auth.json` (or `$CODEX_HOME/auth.json`) to this machine."
    )


def get_codex_usage() -> dict[str, Any]:
    """Fetch the current ChatGPT/Codex usage windows for the OAuth account."""
    token = get_token()
    usage_url = os.environ.get("NANOBOT_CODEX_USAGE_URL", DEFAULT_CODEX_USAGE_URL)
    try:
        response = httpx.get(
            usage_url,
            headers={
                "Authorization": f"Bearer {token.access}",
                "ChatGPT-Account-ID": token.account_id,
                "originator": "codex_cli_rs",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise CodexUsageError("Codex quota request failed due to a network error.") from exc

    if response.status_code == 401:
        raise CodexUsageError("Codex OAuth credentials were rejected. Run the OpenAI login again.")
    if response.status_code == 429:
        raise CodexUsageError("Codex quota endpoint is rate-limited. Please try again shortly.")
    if response.status_code != 200:
        raise CodexUsageError(f"Codex quota request failed (HTTP {response.status_code}).")

    try:
        payload = response.json()
    except ValueError as exc:
        raise CodexUsageError("Codex returned an invalid quota response.") from exc
    if not isinstance(payload, dict):
        raise CodexUsageError("Codex returned an unexpected quota response.")
    return payload


def format_codex_usage(payload: dict[str, Any]) -> str:
    """Render the Codex usage response without exposing token data."""
    plan = _first_value(payload, "plan_type", "planType")
    rate_limit = _first_value(payload, "rate_limit", "rateLimits")
    lines = ["Codex quota" + (f" ({plan})" if plan else "") + ":"]

    if not isinstance(rate_limit, dict):
        lines.append("No rate-limit data was returned.")
        return "\n".join(lines)

    windows = (
        ("Primary", _first_value(rate_limit, "primary_window", "primary")),
        ("Secondary", _first_value(rate_limit, "secondary_window", "secondary")),
    )
    rendered = 0
    for label, window in windows:
        if not isinstance(window, dict):
            continue
        details: list[str] = []
        used = _first_value(window, "used_percent", "usedPercent")
        used_number = _as_number(used)
        if used_number is not None:
            remaining = max(0.0, 100.0 - used_number)
            details.append(f"{used_number:g}% used ({remaining:g}% remaining)")

        duration = _window_duration_seconds(window)
        if duration is not None:
            details.append(f"window {_format_duration(duration)}")

        reset_at = _as_timestamp(_first_value(window, "reset_at", "resetsAt"))
        if reset_at is not None:
            seconds_left = max(0, int(reset_at - time.time()))
            reset_time = datetime.fromtimestamp(reset_at, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            details.append(f"resets in {_format_duration(seconds_left)} ({reset_time})")

        if details:
            lines.append(f"{label}: " + "; ".join(details))
            rendered += 1

    if rendered == 0:
        lines.append("No rate-limit windows were returned.")
    return "\n".join(lines)


def _first_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_timestamp(value: Any) -> float | None:
    timestamp = _as_number(value)
    if timestamp is None:
        return None
    return timestamp / 1000 if timestamp > 10_000_000_000 else timestamp


def _window_duration_seconds(window: dict[str, Any]) -> int | None:
    seconds = _first_value(window, "limit_window_seconds", "window_duration_seconds")
    number = _as_number(seconds)
    if number is not None:
        return max(0, int(number))
    minutes = _as_number(_first_value(window, "window_duration_mins", "windowDurationMins"))
    if minutes is not None:
        return max(0, int(minutes * 60))
    return None


def _format_duration(seconds: int) -> str:
    if seconds >= 3600:
        hours, remainder = divmod(seconds, 3600)
        minutes = remainder // 60
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    minutes, remainder = divmod(seconds, 60)
    if minutes and remainder:
        return f"{minutes}m {remainder}s"
    return f"{minutes}m" if minutes else f"{remainder}s"


def _nanobot_token_path() -> Path:
    override = os.environ.get("OAUTH_CLI_KIT_TOKEN_PATH")
    if override:
        return Path(override)
    xdg_home = os.environ.get("XDG_DATA_HOME")
    if xdg_home:
        base = Path(xdg_home)
    else:
        home = Path(os.environ.get("HOME", str(Path.home())))
        base = home / ".local" / "share"
    return base / "oauth-cli-kit" / "auth" / OPENAI_CODEX_PROVIDER.token_filename


def _official_codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "auth.json"
    home = Path(os.environ.get("HOME", str(Path.home())))
    return home / ".codex" / "auth.json"


def _load_nanobot_token(path: Path) -> OAuthToken | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    access = _clean_str(payload.get("access"))
    refresh = _clean_str(payload.get("refresh"))
    account_id = _clean_str(payload.get("account_id")) or _account_id_from_access_token(access)
    expires = _coerce_expires_ms(payload.get("expires"), access)
    if not access or not account_id or expires is None:
        return None
    return OAuthToken(access=access, refresh=refresh or "", expires=expires, account_id=account_id)


def _load_official_codex_token(path: Path) -> OAuthToken | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    token_block = payload.get("tokens")
    data = token_block if isinstance(token_block, dict) else payload
    access = _clean_str(data.get("access_token"))
    refresh = _clean_str(data.get("refresh_token"))
    account_id = _clean_str(data.get("account_id")) or _account_id_from_access_token(access)
    expires = _coerce_expires_ms(
        data.get("expires") or data.get("expires_at") or payload.get("expires") or payload.get("expires_at"),
        access,
        last_refresh=payload.get("last_refresh"),
    )
    if not access or not account_id or expires is None:
        return None
    return OAuthToken(access=access, refresh=refresh or "", expires=expires, account_id=account_id)


def _save_nanobot_token(path: Path, token: OAuthToken) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "access": token.access,
        "refresh": token.refresh,
        "expires": token.expires,
        "account_id": token.account_id,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _write_official_codex_auth(path: Path, token: OAuthToken) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {}
    tokens["access_token"] = token.access
    tokens["refresh_token"] = token.refresh
    tokens["account_id"] = token.account_id
    payload["tokens"] = tokens
    payload["last_refresh"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _refresh_token(refresh_token: str, account_id: str | None) -> OAuthToken:
    response = httpx.post(
        OPENAI_CODEX_PROVIDER.token_url,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OPENAI_CODEX_PROVIDER.client_id,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Token refresh failed: {response.status_code} {response.text}")

    payload = response.json()
    access = _clean_str(payload.get("access_token"))
    if not access:
        raise RuntimeError("Token refresh response missing access_token")

    refreshed_token = _clean_str(payload.get("refresh_token")) or refresh_token
    refreshed_account = _clean_str(payload.get("account_id")) or account_id or _account_id_from_access_token(access)
    expires = _coerce_expires_ms(payload.get("expires_in") or payload.get("expires_at"), access)
    if not refreshed_account or expires is None:
        raise RuntimeError("Token refresh response missing account metadata")

    return OAuthToken(
        access=access,
        refresh=refreshed_token,
        expires=expires,
        account_id=refreshed_account,
    )


def _coerce_expires_ms(raw: Any, access_token: str | None, last_refresh: Any | None = None) -> int | None:
    if isinstance(raw, int):
        return raw if raw > 10_000_000_000 else _now_ms() + raw * 1000
    if isinstance(raw, float):
        value = int(raw)
        return value if value > 10_000_000_000 else _now_ms() + value * 1000
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if text.isdigit():
            value = int(text)
            return value if value > 10_000_000_000 else _now_ms() + value * 1000
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        if dt is not None:
            return int(dt.timestamp() * 1000)

    exp = _jwt_expiry_ms(access_token)
    if exp is not None:
        return exp

    if isinstance(last_refresh, str):
        try:
            dt = datetime.fromisoformat(last_refresh.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        if dt is not None:
            return int((dt.timestamp() + 3600) * 1000)

    return None


def _jwt_expiry_ms(access_token: str | None) -> int | None:
    payload = _decode_jwt_payload(access_token)
    exp = payload.get("exp")
    if isinstance(exp, int):
        return exp * 1000
    if isinstance(exp, float):
        return int(exp * 1000)
    return None


def _account_id_from_access_token(access_token: str | None) -> str | None:
    payload = _decode_jwt_payload(access_token)
    auth_block = payload.get(OPENAI_CODEX_PROVIDER.jwt_claim_path or "")
    if isinstance(auth_block, dict):
        account_id = _clean_str(auth_block.get(OPENAI_CODEX_PROVIDER.account_id_claim or ""))
        if account_id:
            return account_id
    return None


def _decode_jwt_payload(token: str | None) -> dict[str, Any]:
    if not token:
        return {}
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        padding = "=" * (-len(parts[1]) % 4)
        payload_bytes = base64.urlsafe_b64decode(parts[1] + padding)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def should_refresh_access_token(token: str, threshold_seconds: int = _REFRESH_EARLY_SECONDS) -> bool:
    """Return True when the JWT is close to expiry."""
    exp_ms = _jwt_expiry_ms(token)
    if exp_ms is None:
        return False
    return exp_ms - _now_ms() <= threshold_seconds * 1000
