"""Regression test: agent-workspace/runs/ is a must-not-sync path.

Proves that even when the include patterns contain a broad
``agent-workspace/**`` and ``runs/**`` (from the managed context repo
writable defaults), the explicit ``neverCommit`` exclude rule
``agent-workspace/runs/**`` wins because the sync path/commit validation
gives excludes precedence over includes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from nanobot.context_repo.manager import (
    ContextRepoRuntimeConfig,
    ManagedContextRepo,
)
from nanobot.utils.git_sync import (
    _matches,
    _select_changed_paths,
    _validate_ahead_commits,
    sync_context_repo,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _configure_user(repo: Path) -> None:
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _bootstrap_with_manifest(tmp_path: Path, manifest: dict) -> tuple[Path, Path]:
    """Create a bare remote + local clone whose manifest mirrors the
    hosted staging client-marketing-assistance manifest."""
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    local = tmp_path / "local"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure_user(seed)
    (seed / "nanobot.context.json").write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8"
    )
    (seed / "tools").mkdir(parents=True)
    (seed / "tools" / "common.py").write_text("base\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "branch", "-M", "main")
    _git(seed, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(tmp_path, "clone", str(remote), str(local))
    _configure_user(local)
    return remote, local


# Mirrors the hosted staging manifest's sync section (abridged but with the
# same include/neverCommit semantics).
_STAGING_LIKE_MANIFEST = {
    "name": "client-marketing-assistance",
    "writable": [
        "agent-workspace/**",
        "tools/**",
        "scripts/**",
        "stores/**",
        "modules/**",
        "skills/**",
        "docs/**",
    ],
    "protected": ["config/*.env", "secrets/**", ".nanobot/**"],
    "sync": {
        "autoPush": True,
        "include": [
            "agent-workspace/outputs/**",
            "agent-workspace/feedback/**",
            "agent-workspace/data/marketing-pipeline.json",
            "agent-workspace/data/analytics-pipeline.json",
            "agent-workspace/data/analytics-snapshots.json",
        ],
        "neverCommit": [
            "config/*.env",
            "secrets/**",
            ".nanobot/**",
            "agent-workspace/imports/**",
            "agent-workspace/data/social-comments.json",
            "agent-workspace/runs/**",
            "agent-workspace/state/**",
            "memory/**",
            "sessions/**",
        ],
    },
    "stores": {
        "marketing_pipeline": {
            "path": "agent-workspace/data/marketing-pipeline.json",
            "syncPaths": ["agent-workspace/data/marketing-pipeline.json"],
        },
        "analytics_pipeline": {
            "path": "agent-workspace/data/analytics-pipeline.json",
            "syncPaths": ["agent-workspace/data/analytics-pipeline.json"],
        },
        "analytics_snapshots": {
            "path": "agent-workspace/data/analytics-snapshots.json",
            "syncPaths": ["agent-workspace/data/analytics-snapshots.json"],
        },
        "social_comments": {
            "path": "agent-workspace/data/social-comments.json",
            "syncPaths": ["agent-workspace/data/social-comments.json"],
        },
    },
}


def test_effective_policy_excludes_agent_workspace_runs(tmp_path: Path) -> None:
    remote, local = _bootstrap_with_manifest(tmp_path, _STAGING_LIKE_MANIFEST)
    config = ContextRepoRuntimeConfig.from_raw({"path": str(local), "autoSync": True})
    repo = ManagedContextRepo.load(config)
    include = repo.sync_include_patterns()
    exclude = repo.sync_exclude_patterns()

    # The broad include patterns are present...
    assert _matches("agent-workspace/runs/test.json", include)
    assert _matches("runs/test.json", include)
    # ...but the explicit neverCommit exclude also matches.
    assert _matches("agent-workspace/runs/test.json", exclude)
    # Exclude wins for the combined decision.
    assert _select_changed_paths(
        local,
        include_paths=include,
        exclude_paths=exclude,
    ) == []

    # Even a pre-committed runs file is refused by the ahead-commit audit.
    runs_dir = local / "agent-workspace" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "test.json").write_text('{"a": 1}\n', encoding="utf-8")
    _git(local, "add", "agent-workspace/runs/test.json")
    _git(local, "commit", "-m", "runs leak attempt")

    ok, errors = _validate_ahead_commits(
        local,
        upstream="origin/main",
        include_paths=include,
        exclude_paths=exclude,
    )
    assert ok is False
    assert any("prohibited path agent-workspace/runs/test.json" in e for e in errors)

    # And sync_context_repo refuses the push.
    assert sync_context_repo(
        local,
        include_paths=include,
        exclude_paths=exclude,
        message="nanobot: sync test",
    ) is False
