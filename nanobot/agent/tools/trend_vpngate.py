"""Agent-side VPN Gate relay discovery for trend research.

The API response and downloaded profiles are untrusted input. This module
only ranks a small candidate set; the privileged broker validates every
profile and proves the tunnel works before it creates a session.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import ipaddress
import re
import ssl
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://www.vpngate.net/api/iphone/"
MAX_API_BYTES = 32 * 1024 * 1024
# The broker's JSON-lines request is bounded at 32 KiB. Leave room for JSON
# escaping and relay metadata when sending an inline profile.
MAX_PROFILE_BYTES = 12 * 1024
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")


class VpnGateDiscoveryError(RuntimeError):
    """Raised when the official VPN Gate list cannot produce candidates."""


@dataclass(frozen=True)
class VpnGateCandidate:
    hostname: str
    ip_address: str
    score: int
    ping_ms: int | None
    speed_bps: int
    country: str
    sessions: int
    profile_text: str

    @property
    def label(self) -> str:
        country = self.country or "unknown country"
        return f"VPN Gate {self.hostname} ({country})"


def _integer(value: str, *, default: int = 0) -> int:
    try:
        parsed = int(value.strip())
    except (AttributeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _ping(value: str) -> int | None:
    parsed = _integer(value, default=0)
    return parsed or None


def _is_global_ipv4(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return False
    return address.version == 4 and address.is_global


def _is_safe_hostname(value: str) -> bool:
    host = value.strip().rstrip(".").lower()
    return bool(
        _HOSTNAME_RE.fullmatch(host)
        and ".." not in host
        and (host.endswith(".opengw.net") or "." not in host)
    )


def _decode_profile(encoded: str) -> str:
    encoded = re.sub(r"\s+", "", encoded)
    if not encoded or len(encoded) > MAX_PROFILE_BYTES * 2:
        raise VpnGateDiscoveryError("VPN Gate profile is missing or too large")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise VpnGateDiscoveryError("VPN Gate profile is not valid base64") from exc
    if len(decoded) > MAX_PROFILE_BYTES:
        raise VpnGateDiscoveryError("VPN Gate profile is too large for the broker protocol")
    try:
        profile = decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VpnGateDiscoveryError("VPN Gate profile is not UTF-8") from exc
    if "\x00" in profile or not re.search(r"(?m)^\s*remote\s+\S+", profile):
        raise VpnGateDiscoveryError("VPN Gate profile is malformed")
    return profile


def parse_api_response(text: str, *, max_candidates: int = 8) -> list[VpnGateCandidate]:
    """Parse the official CSV API and return ranked, bounded candidates."""

    if not isinstance(text, str) or not text.strip():
        raise VpnGateDiscoveryError("VPN Gate returned an empty relay list")
    if len(text.encode("utf-8", errors="replace")) > MAX_API_BYTES:
        raise VpnGateDiscoveryError("VPN Gate relay list is too large")
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool):
        raise VpnGateDiscoveryError("max_candidates must be an integer")
    if not 1 <= max_candidates <= 16:
        raise VpnGateDiscoveryError("max_candidates must be between 1 and 16")

    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None
    rows: list[list[str]] = []
    for row in reader:
        if not row:
            continue
        first = row[0].strip()
        if first == "*vpn_servers":
            continue
        if first.startswith("*vpn_servers"):
            suffix = first[len("*vpn_servers") :].strip()
            if suffix.startswith("#"):
                possible_header = [suffix[1:].strip(), *[item.strip() for item in row[1:]]]
                if "OpenVPN_ConfigData_Base64" in possible_header:
                    header = possible_header
            continue
        if first.startswith("#"):
            possible_header = [first[1:].strip(), *[item.strip() for item in row[1:]]]
            if "OpenVPN_ConfigData_Base64" in possible_header:
                header = possible_header
            continue
        if header is not None:
            rows.append(row)

    if not header:
        raise VpnGateDiscoveryError("VPN Gate relay list has no recognized CSV header")
    indexes = {name: index for index, name in enumerate(header)}
    required = {"HostName", "IP", "Score", "Ping", "Speed", "CountryLong"}
    if not required.issubset(indexes) or "OpenVPN_ConfigData_Base64" not in indexes:
        raise VpnGateDiscoveryError("VPN Gate relay list is missing required columns")

    candidates: list[VpnGateCandidate] = []
    for row in rows:
        required_indexes = [indexes[name] for name in required]
        config_index = indexes["OpenVPN_ConfigData_Base64"]
        if max([*required_indexes, config_index], default=0) >= len(row):
            continue
        hostname = row[indexes["HostName"]].strip().rstrip(".").lower()
        ip_address = row[indexes["IP"]].strip()
        if not _is_safe_hostname(hostname) or not _is_global_ipv4(ip_address):
            continue
        try:
            profile = _decode_profile(row[config_index])
        except VpnGateDiscoveryError:
            continue
        sessions_index = indexes.get("NumVpnSessions")
        sessions = (
            _integer(row[sessions_index])
            if sessions_index is not None and sessions_index < len(row)
            else 0
        )
        candidates.append(
            VpnGateCandidate(
                hostname=hostname,
                ip_address=ip_address,
                score=_integer(row[indexes["Score"]]),
                ping_ms=_ping(row[indexes["Ping"]]),
                speed_bps=_integer(row[indexes["Speed"]]),
                country=row[indexes["CountryLong"]].strip()[:80],
                sessions=sessions,
                profile_text=profile,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.score,
            item.ping_ms if item.ping_ms is not None else 999999,
            -item.speed_bps,
            item.sessions,
            item.hostname,
        )
    )
    selected: list[VpnGateCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.hostname, candidate.ip_address)
        if key in seen:
            continue
        selected.append(candidate)
        seen.add(key)
        if len(selected) >= max_candidates:
            break
    if not selected:
        raise VpnGateDiscoveryError("VPN Gate returned no usable OpenVPN profiles")
    return selected


def fetch_candidates(
    *, max_candidates: int = 8, timeout_seconds: float = 15.0
) -> list[VpnGateCandidate]:
    """Fetch the official relay list directly from the agent process."""

    request = Request(
        DEFAULT_API_URL,
        headers={"Accept": "text/plain", "User-Agent": "marketing-agent-trend-research/1"},
        method="GET",
    )
    try:
        with urlopen(
            request, timeout=timeout_seconds, context=ssl.create_default_context()
        ) as response:
            body = response.read(MAX_API_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise VpnGateDiscoveryError(f"VPN Gate relay list fetch failed: {exc}") from exc
    if len(body) > MAX_API_BYTES:
        raise VpnGateDiscoveryError("VPN Gate relay list is too large")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VpnGateDiscoveryError("VPN Gate relay list is not UTF-8") from exc
    return parse_api_response(text, max_candidates=max_candidates)


__all__ = [
    "DEFAULT_API_URL",
    "VpnGateCandidate",
    "VpnGateDiscoveryError",
    "fetch_candidates",
    "parse_api_response",
]
