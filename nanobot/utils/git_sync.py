"""Git sync utility for context repositories."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from fnmatch import fnmatch
import os
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.secret_scan import find_secrets_in_blob


@dataclass(frozen=True)
class _PathSnapshot:
    rel_path: str
    content: bytes | None


def _run_git(repo: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run a git command in the given repo directory."""
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=10")
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _git_output(result: subprocess.CompletedProcess[str]) -> str:
    """Return compact stdout/stderr text for logs."""
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    try:
        result = _run_git(path, "rev-parse", "--is-inside-work-tree")
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def has_unmerged_paths(repo: Path) -> bool:
    """Check if the repo has unresolved merge/rebase conflicts."""
    try:
        result = _run_git(repo, "diff", "--name-only", "--diff-filter=U")
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True


def has_changes(repo: Path) -> bool:
    """Check if the git repo has uncommitted changes."""
    try:
        result = _run_git(repo, "status", "--porcelain", "--untracked-files=all")
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _changed_paths(repo: Path) -> list[str]:
    """Return changed paths relative to repo root, including untracked files."""
    try:
        result = _run_git(repo, "status", "--porcelain")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if result.returncode != 0:
        return []
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            raw = raw.rsplit(" -> ", 1)[1]
        raw = raw.strip('"')
        if raw:
            if raw.endswith("/") and (repo / raw).is_dir():
                for child in sorted((repo / raw).rglob("*")):
                    if child.is_file():
                        paths.append(child.relative_to(repo).as_posix())
            else:
                paths.append(raw)
    return paths


def _matches(path: str, patterns: list[str]) -> bool:
    clean = path.strip("/")
    for pattern in patterns:
        pat = pattern.strip().strip("/")
        if not pat:
            continue
        if pat.endswith("/**"):
            base = pat[:-3].strip("/")
            if clean == base or clean.startswith(base + "/"):
                return True
        if fnmatch(clean, pat):
            return True
    return False


def _select_changed_paths(
    repo: Path,
    include_paths: list[str] | None,
    exclude_paths: list[str] | None,
) -> list[str]:
    changed = _changed_paths(repo)
    if include_paths is None and not exclude_paths:
        return changed
    selected: list[str] = []
    for path in changed:
        if include_paths is not None and not _matches(path, include_paths):
            continue
        if exclude_paths and _matches(path, exclude_paths):
            continue
        selected.append(path)
    return selected


def _snapshot_paths(repo: Path, paths: list[str]) -> list[_PathSnapshot]:
    snapshots: list[_PathSnapshot] = []
    for rel_path in paths:
        path = repo / rel_path
        if path.exists() and path.is_file():
            snapshots.append(_PathSnapshot(rel_path=rel_path, content=path.read_bytes()))
        else:
            snapshots.append(_PathSnapshot(rel_path=rel_path, content=None))
    return snapshots


def _restore_snapshots(repo: Path, snapshots: list[_PathSnapshot]) -> None:
    for snapshot in snapshots:
        path = repo / snapshot.rel_path
        if snapshot.content is None:
            if path.exists() and path.is_file():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(snapshot.content)


def _upstream_ref(repo: Path) -> str | None:
    result = _run_git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    if result.returncode != 0:
        logger.error("Context repo upstream lookup failed: {}", _git_output(result))
        return None
    ref = result.stdout.strip()
    return ref or None


def _commit_selected_paths(repo: Path, selected_paths: list[str], message: str) -> bool:
    if not selected_paths:
        return True

    add = _run_git(repo, "add", "-A", "--", *selected_paths)
    if add.returncode != 0:
        logger.error("Context repo git add failed: {}", _git_output(add))
        return False

    commit = _run_git(repo, "commit", "-m", message)
    if commit.returncode != 0:
        output = _git_output(commit)
        if "nothing to commit" in output:
            return True
        logger.error("Context repo git commit failed: {}", output)
        return False
    return True


def sync_with_remote(repo: Path) -> bool:
    """Fetch and rebase onto the configured upstream before local commits/pushes."""
    if has_unmerged_paths(repo):
        logger.error("Context repo has unresolved git conflicts; refusing to sync: {}", repo)
        return False

    fetch = _run_git(repo, "fetch", "--prune", timeout=60)
    if fetch.returncode != 0:
        logger.error("Context repo git fetch failed: {}", _git_output(fetch))
        return False

    pull = _run_git(repo, "pull", "--rebase", "--autostash", timeout=60)
    if pull.returncode != 0:
        logger.error("Context repo git pull --rebase failed: {}", _git_output(pull))
        _abort_interrupted_rebase(repo)
        return False

    if has_unmerged_paths(repo):
        logger.error("Context repo has unresolved conflicts after pull; refusing to sync: {}", repo)
        _abort_interrupted_rebase(repo)
        return False
    return True


def sync_with_remote_reapplying_changes(
    repo: Path,
    *,
    selected_paths: list[str],
    all_changed_paths: list[str],
    snapshots: list[_PathSnapshot],
) -> bool:
    """Sync with remote and recover conflicts by replaying selected local edits."""
    if sync_with_remote(repo):
        return True

    if not selected_paths:
        return False

    unselected = sorted(set(all_changed_paths) - set(selected_paths))
    if unselected:
        logger.error(
            "Context repo sync conflict needs developer attention; unselected local changes would be at risk: {}",
            ", ".join(unselected),
        )
        return False

    _abort_interrupted_rebase(repo)

    fetch = _run_git(repo, "fetch", "--prune", timeout=60)
    if fetch.returncode != 0:
        logger.error("Context repo git fetch failed during recovery: {}", _git_output(fetch))
        return False

    upstream = _upstream_ref(repo)
    if not upstream:
        return False

    reset = _run_git(repo, "reset", "--hard", upstream, timeout=60)
    if reset.returncode != 0:
        logger.error("Context repo git reset recovery failed: {}", _git_output(reset))
        return False

    try:
        _restore_snapshots(repo, snapshots)
    except OSError as exc:
        logger.error("Context repo snapshot restore failed during recovery: {}", exc)
        return False

    if has_unmerged_paths(repo):
        logger.error("Context repo still has unresolved conflicts after recovery: {}", repo)
        return False
    logger.info("Context repo recovered from sync conflict by replaying selected changes: {}", repo)
    return True


def _abort_interrupted_rebase(repo: Path) -> None:
    """Return the repo to its pre-rebase state when an autonomous pull conflicts."""
    git_dir = repo / ".git"
    if not ((git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists()):
        return
    abort = _run_git(repo, "rebase", "--abort", timeout=30)
    if abort.returncode != 0:
        logger.error("Context repo git rebase --abort failed: {}", _git_output(abort))


def _ahead_commit_summaries(repo: Path, upstream: str) -> list[dict[str, Any]]:
    """Return one summary per commit in upstream..HEAD with hash and paths.

    Each item has keys: ``sha`` (full), ``short`` (7-char), ``paths`` (list of
    relative paths touched by the commit), ``subject``.
    """
    fmt = "%H%x1f%h%x1f%s%x1f"
    result = _run_git(
        repo,
        "log",
        f"{upstream}..HEAD",
        "--pretty=format:" + fmt,
        "--name-only",
        timeout=60,
    )
    if result.returncode != 0:
        logger.error("Context repo ahead-commit log failed: {}", _git_output(result))
        return []
    summaries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in result.stdout.splitlines():
        if not line:
            continue
        if "\x1f" in line:
            if current is not None:
                summaries.append(current)
            sha, short, subject = line.split("\x1f", 2)
            current = {
                "sha": sha.strip(),
                "short": short.strip(),
                "subject": subject.strip(),
                "paths": [],
            }
        else:
            if current is not None:
                p = line.strip()
                if p:
                    current["paths"].append(p)
    if current is not None:
        summaries.append(current)
    return summaries


def _commit_blob(repo: Path, sha: str, path: str) -> str | None:
    """Return the blob text for one path in a given commit, or None if absent."""
    result = _run_git(repo, "show", f"{sha}:{path}", timeout=30)
    if result.returncode != 0:
        return None
    return result.stdout


def _validate_ahead_commits(
    repo: Path,
    *,
    upstream: str,
    include_paths: list[str] | None,
    exclude_paths: list[str] | None,
) -> tuple[bool, list[str]]:
    """Inspect every local-ahead commit. Return (ok, error_lines).

    Each commit must:
        1. Touch only paths that match the include list (or all paths if
           include is None) AND do not match the exclude list.
        2. Have blob content free of probable credentials.

    Error lines describe ONLY the offending path, commit short-hash, and
    redacted fingerprint. Full credential values are never written.
    """
    summaries = _ahead_commit_summaries(repo, upstream)
    errors: list[str] = []
    for summary in summaries:
        commit = summary["short"]
        # 1. Path policy
        for path in summary["paths"]:
            allowed = (
                include_paths is None
                or _matches(path, include_paths)
            )
            excluded = bool(exclude_paths) and _matches(path, exclude_paths)
            if not allowed or excluded:
                errors.append(
                    f"commit {commit} contains prohibited path {path}"
                )
                # No need to scan the blob if the path is already disallowed.
                continue
            # 2. Secret content policy
            blob = _commit_blob(repo, summary["sha"], path)
            if blob is None:
                continue
            for hit in find_secrets_in_blob(blob):
                errors.append(
                    "commit {commit} contains probable {kind} credential "
                    "in {path} (fingerprint: {fp}, field: {field})".format(
                        commit=commit,
                        kind=hit["kind"],
                        path=path,
                        fp=hit["fingerprint"],
                        field=hit["field"],
                    )
                )
    return (len(errors) == 0, errors)


def sync_context_repo(
    repo: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    message: str = "nanobot: auto-sync context updates",
) -> bool:
    """Commit and push any changes in the context repo.

    Returns True if sync succeeded (or nothing to sync), False on failure.

    The push safety check is based on the actual content and path list of the
    local-ahead commits (upstream..HEAD), not on the current working-tree
    state. Once a managed change has been committed, an empty working-tree
    ``selected_paths`` is normal and must NOT block the push of a valid
    commit.
    """
    if not is_git_repo(repo):
        logger.debug("Context path is not a git repo, skipping sync: {}", repo)
        return False

    all_changed_paths = _changed_paths(repo)
    selected_paths = _select_changed_paths(repo, include_paths, exclude_paths)
    snapshots = _snapshot_paths(repo, selected_paths)

    if not sync_with_remote_reapplying_changes(
        repo,
        selected_paths=selected_paths,
        all_changed_paths=all_changed_paths,
        snapshots=snapshots,
    ):
        return False

    if not has_changes(repo):
        logger.debug("No working-tree changes in context repo: {}", repo)

    selected_paths = _select_changed_paths(repo, include_paths, exclude_paths)

    try:
        # Commit managed working-tree changes (if any) so the local-ahead
        # range covers them. The ahead-commit validation below then applies
        # the same policy to the resulting ahead commits.
        if selected_paths and not _commit_selected_paths(
            repo, selected_paths, message
        ):
            return False

        # The push safety decision is now based on the actual ahead-commit
        # content + path policy, evaluated AFTER each sync_with_remote attempt
        # so remote advancement is re-validated. This runs whether or not
        # the working tree had a change to commit -- an operator may have
        # made a local commit on a different machine and a prior sync run
        # left an unsafe commit unpushed.
        upstream = _upstream_ref(repo)
        if not upstream:
            return False

        # If the working tree was clean, there are no ahead commits either
        # (otherwise the previous sync would have either pushed them or
        # refused the push). In that case, there is nothing to do.
        if not selected_paths:
            # Verify there are no pre-existing ahead commits from prior runs.
            existing = _ahead_commit_summaries(repo, upstream)
            if not existing:
                logger.debug("No selected changes in context repo: {}", repo)
                return True
            # Fall through: existing ahead commits still need to be pushed
            # (or refused) per the safety policy.

        for attempt in range(4):
            # Re-sync with remote, then recompute and re-validate ahead commits.
            if not sync_with_remote_reapplying_changes(
                repo,
                selected_paths=selected_paths,
                all_changed_paths=_changed_paths(repo) or all_changed_paths,
                snapshots=_snapshot_paths(repo, selected_paths),
            ):
                return False

            # The recovery path inside sync_with_remote_reapplying_changes may
            # have left managed files dirty on disk (after a rebase conflict
            # was reset and snapshots were restored). Re-commit them so the
            # local-ahead range is non-empty for the safety check below. This
            # preserves the unselected dirty files: only the managed files
            # are staged and committed.
            current_selected = _select_changed_paths(
                repo, include_paths, exclude_paths
            )
            if current_selected and not _commit_selected_paths(
                repo, current_selected, message
            ):
                return False

            ok, errors = _validate_ahead_commits(
                repo,
                upstream=upstream,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
            )
            if not ok:
                for line in errors:
                    logger.error("Context repo push refused: {}", line)
                return False

            push = _run_git(repo, "push", timeout=60)
            if push.returncode == 0:
                logger.info("Context repo synced successfully: {}", repo)
                return True
            logger.warning(
                "Context repo push attempt {} failed: {}",
                attempt + 1,
                _git_output(push),
            )
            if attempt < 3:
                import time
                time.sleep(2 ** (attempt + 1))

        logger.error("Context repo push failed after 4 attempts")
        return False

    except subprocess.TimeoutExpired:
        logger.error("Context repo sync timed out: {}", repo)
        return False
    except FileNotFoundError:
        logger.error("git not found on PATH, cannot sync context repo")
        return False


async def async_sync_context_repo(
    repo: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    message: str = "nanobot: auto-sync context updates",
) -> bool:
    """Run sync_context_repo in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: sync_context_repo(
            repo,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            message=message,
        ),
    )
