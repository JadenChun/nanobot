"""Capability profiles for foreground delegated roles.

The main agent may expose a broad tool registry, but a delegated runner gets a
separate registry built from one of these immutable profiles.  Keeping the
matrix in a small dependency-free module makes accidental capability widening
easy to detect in tests and avoids using prompts as an authorization boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

FILE_READ_CAPABILITIES = frozenset({"read_file", "list_dir", "search_files"})
WEB_READ_CAPABILITIES = frozenset({"web_search", "web_fetch"})

PLANNER_CAPABILITIES = FILE_READ_CAPABILITIES | WEB_READ_CAPABILITIES
REVIEWER_CAPABILITIES = PLANNER_CAPABILITIES | frozenset({"agent_browser"})
EXPLORER_CAPABILITIES = REVIEWER_CAPABILITIES | frozenset({"agent_device"})
CRAWLER_CAPABILITIES = frozenset({"social_crawl"})
WORKER_CAPABILITIES = frozenset({
    "read_file",
    "list_dir",
    "write_file",
    "edit_file",
})


@dataclass(frozen=True, slots=True)
class RoleCapabilityProfile:
    """Static capability contract for one delegated role."""

    role: str
    tools: frozenset[str]
    read_only: bool
    allow_read_only_mcp: bool = False


ROLE_CAPABILITIES = MappingProxyType({
    "planner": RoleCapabilityProfile(
        role="planner",
        tools=PLANNER_CAPABILITIES,
        read_only=True,
        allow_read_only_mcp=True,
    ),
    "reviewer": RoleCapabilityProfile(
        role="reviewer",
        tools=REVIEWER_CAPABILITIES,
        read_only=True,
        allow_read_only_mcp=True,
    ),
    "explorer": RoleCapabilityProfile(
        role="explorer",
        tools=EXPLORER_CAPABILITIES,
        read_only=True,
        allow_read_only_mcp=True,
    ),
    "crawler": RoleCapabilityProfile(
        role="crawler",
        tools=CRAWLER_CAPABILITIES,
        read_only=True,
        allow_read_only_mcp=False,
    ),
    "worker": RoleCapabilityProfile(
        role="worker",
        tools=WORKER_CAPABILITIES,
        read_only=False,
        allow_read_only_mcp=False,
    ),
})

# Convenient aliases for callers that describe the registry rather than the
# profile object.  These are immutable so a caller cannot elevate another
# role by mutating shared state.
ROLE_TOOL_SETS = MappingProxyType({
    role: profile.tools for role, profile in ROLE_CAPABILITIES.items()
})
DELEGATED_READ_ONLY_ROLES = frozenset({"planner", "reviewer", "explorer", "crawler"})


def role_capabilities(role: str) -> RoleCapabilityProfile:
    """Return the exact profile for *role*, failing closed for unknown roles."""

    normalized = role.strip().lower() if isinstance(role, str) else ""
    try:
        return ROLE_CAPABILITIES[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown delegated role: {role!r}") from exc


def role_tool_names(
    role: str,
    *,
    read_only_mcp_tools: Iterable[str] = (),
) -> frozenset[str]:
    """Return the profile's tools plus explicitly classified read-only MCP tools."""

    profile = role_capabilities(role)
    if not profile.allow_read_only_mcp:
        return profile.tools
    return profile.tools | frozenset(
        name for name in read_only_mcp_tools if isinstance(name, str) and name
    )


__all__ = [
    "CRAWLER_CAPABILITIES",
    "DELEGATED_READ_ONLY_ROLES",
    "EXPLORER_CAPABILITIES",
    "FILE_READ_CAPABILITIES",
    "PLANNER_CAPABILITIES",
    "REVIEWER_CAPABILITIES",
    "ROLE_CAPABILITIES",
    "ROLE_TOOL_SETS",
    "RoleCapabilityProfile",
    "WEB_READ_CAPABILITIES",
    "WORKER_CAPABILITIES",
    "role_capabilities",
    "role_tool_names",
]
