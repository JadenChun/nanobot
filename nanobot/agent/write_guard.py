"""Shared write-scope and file-lock helpers for agent mutations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WriteScope:
    """A normalized writable target rooted in the current workspace."""

    path: Path
    recursive: bool = False

    @classmethod
    def from_raw(cls, workspace: Path, raw: str) -> "WriteScope":
        value = (raw or "").strip()
        if not value:
            raise ValueError("write_scope entries must not be empty")

        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            raise ValueError("write_scope entries must be workspace-relative paths")

        recursive = value.endswith(("/", "\\"))
        normalized = value.rstrip("/\\") if recursive else value
        if not normalized:
            raise ValueError("write_scope directory prefixes must include a path before the trailing slash")

        workspace_root = workspace.resolve()
        resolved = (workspace_root / normalized).resolve()
        if not _is_under(resolved, workspace_root):
            raise ValueError("write_scope entries must stay inside the workspace")
        return cls(path=resolved, recursive=recursive)

    def allows(self, target: Path) -> bool:
        resolved = target.resolve()
        if self.recursive:
            return _is_under(resolved, self.path)
        return resolved == self.path

    def overlaps(self, other: "WriteScope") -> bool:
        if self.recursive and other.recursive:
            return _is_under(self.path, other.path) or _is_under(other.path, self.path)
        if self.recursive:
            return self.allows(other.path)
        if other.recursive:
            return other.allows(self.path)
        return self.path == other.path

    def describe(self) -> str:
        text = str(self.path)
        return f"{text}/" if self.recursive else text


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory.resolve())
        return True
    except ValueError:
        return False


class FileLockRegistry:
    """Fail-fast in-process lock registry for file mutations."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._owners: dict[Path, str] = {}

    async def acquire(self, path: Path, owner: str) -> str | None:
        resolved = path.resolve()
        async with self._guard:
            current = self._owners.get(resolved)
            if current is not None:
                return current
            self._owners[resolved] = owner
            return None

    async def release(self, path: Path, owner: str) -> None:
        resolved = path.resolve()
        async with self._guard:
            current = self._owners.get(resolved)
            if current == owner:
                self._owners.pop(resolved, None)
