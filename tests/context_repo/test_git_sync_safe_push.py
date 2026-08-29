"""New regression tests for the rewritten sync_context_repo.

These tests cover the new contract:

- A managed change is committed and pushed successfully (Test H).
- Valid managed commit + unrelated dirty excluded files: push still succeeds,
  unselected dirty files are preserved (Test I).
- A prohibited path in an ahead commit: push is refused with a precise,
  redacted error message (Test J).
- An allowed analytics file containing a credential: push is refused
  (Test K).
- The secret-detector fingerprint in error messages is redacted (Test L).
- A remote push while local commits are pending: revalidate after each
  sync (Test M).
- No ahead commits and no selected changes: no-op success (Test N).
- Local branch already ahead with a valid managed commit: existing
  commit is pushed even if no selected working-tree changes exist
  (Test O).
- Unselected dirty data files survive the operation byte-equivalent
  (Test P).
- A legitimate code commit under tools/ or modules/ or tests/ passes
  the path policy (Test Q).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from nanobot.utils import git_sync
from nanobot.utils.git_sync import sync_context_repo


# ---------------------------------------------------------------------------
# Helpers (mirror the existing test_git_sync.py style)
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _git_out(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _configure_user(repo: Path) -> None:
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _bootstrap_remote_with_initial(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    """Create bare remote, seed with one initial commit, return (remote, seed, local)."""
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    local = tmp_path / "local"

    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure_user(seed)
    (seed / "tools").mkdir(parents=True)
    (seed / "tools" / "common.py").write_text("base\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "branch", "-M", "main")
    _git(seed, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(tmp_path, "clone", str(remote), str(local))
    _configure_user(local)
    return remote, seed, local


# ---------------------------------------------------------------------------
# Test H: normal managed change -> commit -> valid ahead commit -> push succeeds
# ---------------------------------------------------------------------------


def test_h_normal_managed_change_pushes(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)

    (local / "tools" / "new.py").write_text("print('new')\n", encoding="utf-8")

    assert sync_context_repo(
        local,
        include_paths=["tools/**"],
        exclude_paths=[],
        message="nanobot: sync test",
    )

    verify = tmp_path / "verify"
    _git(tmp_path, "clone", str(remote), str(verify))
    assert (verify / "tools" / "new.py").read_text(encoding="utf-8") == "print('new')\n"


# ---------------------------------------------------------------------------
# Test I: valid managed commit + unrelated dirty excluded files
# ---------------------------------------------------------------------------


def test_i_unrelated_dirty_excluded_files_preserved(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)

    (local / "tools" / "managed.py").write_text("managed\n", encoding="utf-8")
    # Excluded runtime data the operator does NOT want synced.
    (local / "agent-workspace" / "state").mkdir(parents=True)
    runtime_state = local / "agent-workspace" / "state" / "sync-state.json"
    runtime_state.write_text('{"timestamp": "2026-08-29T00:00:00Z"}', encoding="utf-8")
    runtime_state_content_before = runtime_state.read_bytes()

    assert sync_context_repo(
        local,
        include_paths=["tools/**"],
        exclude_paths=["agent-workspace/state/**"],
        message="nanobot: sync test",
    )

    # Managed change was pushed.
    verify = tmp_path / "verify"
    _git(tmp_path, "clone", str(remote), str(verify))
    assert (verify / "tools" / "managed.py").read_text(encoding="utf-8") == "managed\n"

    # Excluded runtime state file survived byte-equivalent.
    assert runtime_state.read_bytes() == runtime_state_content_before


# ---------------------------------------------------------------------------
# Test J: prohibited path in ahead commit -> push refused
# ---------------------------------------------------------------------------


def test_j_prohibited_path_in_ahead_commit_refuses_push(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)
    # Pre-commit a prohibited path manually, then run sync.
    (local / "agent-workspace" / "runs").mkdir(parents=True)
    (local / "agent-workspace" / "runs" / "uat.json").write_text('{"a": 1}\n', encoding="utf-8")
    _git(local, "add", "agent-workspace/runs/uat.json")
    _git(local, "commit", "-m", "pre-existing UAT leak")

    ok = sync_context_repo(
        local,
        include_paths=["tools/**"],
        exclude_paths=["agent-workspace/runs/**"],
        message="nanobot: sync test",
    )
    assert ok is False
    # The remote must NOT contain the prohibited file.
    verify = tmp_path / "verify"
    _git(tmp_path, "clone", str(remote), str(verify))
    assert not (verify / "agent-workspace" / "runs" / "uat.json").exists()


# ---------------------------------------------------------------------------
# Test K: allowed analytics file containing a credential -> push refused
# ---------------------------------------------------------------------------


def test_k_secret_in_allowed_analytics_file_refuses_push(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)

    # Allowed analytics file (a Meta-style data store path that the
    # allow-list permits) but its blob contains a Facebook EAA token.
    analytics_path = local / "agent-workspace" / "data" / "analytics-pipeline.json"
    analytics_path.parent.mkdir(parents=True)
    analytics_path.write_text(
        '{"records": [{"id": "1", "paging": {"next": "https://graph.facebook.com/x?access_token=EAA'
        + "X" * 200
        + '"}}]}\n',
        encoding="utf-8",
    )
    _git(local, "add", "agent-workspace/data/analytics-pipeline.json")
    _git(local, "commit", "-m", "leaked token in analytics")

    ok = sync_context_repo(
        local,
        include_paths=["agent-workspace/data/**"],
        exclude_paths=[],
        message="nanobot: sync test",
    )
    assert ok is False
    # The remote must NOT contain the analytics file with the token.
    verify = tmp_path / "verify"
    _git(tmp_path, "clone", str(remote), str(verify))
    verify_analytics = verify / "agent-workspace" / "data" / "analytics-pipeline.json"
    if verify_analytics.exists():
        content = verify_analytics.read_text(encoding="utf-8")
        assert "EAA" not in content


# ---------------------------------------------------------------------------
# Test L: secret message redaction in error output
# ---------------------------------------------------------------------------


def test_l_secret_error_messages_are_redacted(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)

    analytics_path = local / "agent-workspace" / "data" / "analytics-pipeline.json"
    analytics_path.parent.mkdir(parents=True)
    full_token = "EAA" + "Z" * 200
    analytics_path.write_text(
        '{"url": "https://graph.facebook.com/x?access_token=' + full_token + '"}\n',
        encoding="utf-8",
    )
    _git(local, "add", "agent-workspace/data/analytics-pipeline.json")
    _git(local, "commit", "-m", "leak")

    # Capture stderr from sync_context_repo by patching the loguru logger.
    captured: list[str] = []

    def _capture(message: str) -> None:  # type: ignore[no-untyped-def]
        captured.append(str(message))

    from loguru import logger

    logger.remove()
    logger.add(_capture, level="ERROR")

    ok = sync_context_repo(
        local,
        include_paths=["agent-workspace/data/**"],
        exclude_paths=[],
        message="nanobot: sync test",
    )
    assert ok is False
    full = "\n".join(captured)
    # The full token value must NEVER appear in any error log.
    assert full_token not in full
    # But a redacted fingerprint should be reported.
    assert "EAA" in full and "..." in full


# ---------------------------------------------------------------------------
# Test M: remote advances concurrently -> reconcile, revalidate, push safely
# ---------------------------------------------------------------------------


def test_m_remote_advances_concurrently_is_revalidated(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)
    other = tmp_path / "other"
    _git(tmp_path, "clone", str(remote), str(other))
    _configure_user(other)

    (local / "tools" / "managed.py").write_text("local managed\n", encoding="utf-8")

    pushed_remote_update = False
    original_run_git = git_sync._run_git

    def run_git_with_remote_race(repo: Path, *args: str, timeout: int = 30):
        nonlocal pushed_remote_update
        if (
            repo == local
            and args
            and args[0] == "push"
            and not pushed_remote_update
        ):
            # Remote advances just before our first push attempt.
            (other / "tools" / "managed.py").write_text("remote race\n", encoding="utf-8")
            _git(other, "add", "tools/managed.py")
            _git(other, "commit", "-m", "remote race")
            _git(other, "push")
            pushed_remote_update = True
        return original_run_git(repo, *args, timeout=timeout)

    git_sync._run_git = run_git_with_remote_race  # type: ignore[assignment]
    try:
        assert sync_context_repo(
            local,
            include_paths=["tools/**"],
            exclude_paths=[],
            message="nanobot: sync test",
        )
    finally:
        git_sync._run_git = original_run_git  # type: ignore[assignment]

    assert pushed_remote_update is True
    # Local edit wins after the reconciling rebase + replay + revalidation.
    verify = tmp_path / "verify"
    _git(tmp_path, "clone", str(remote), str(verify))
    assert (verify / "tools" / "managed.py").read_text(encoding="utf-8") == "local managed\n"


# ---------------------------------------------------------------------------
# Test N: no ahead commits and no selected changes -> no-op success
# ---------------------------------------------------------------------------


def test_n_no_changes_no_ahead_commits_is_noop(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)

    # Working tree is clean; no ahead commits.
    assert _git_out(local, "log", "origin/main..HEAD", "--oneline") == ""
    assert sync_context_repo(
        local,
        include_paths=["tools/**"],
        exclude_paths=[],
        message="nanobot: sync test",
    )


# ---------------------------------------------------------------------------
# Test O: branch already ahead with a valid managed commit -> push succeeds
# ---------------------------------------------------------------------------


def test_o_branch_already_ahead_with_valid_commit_pushes(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)

    # Pre-commit a managed change directly (operator-initiated).
    (local / "tools" / "precommit.py").write_text("# operator change\n", encoding="utf-8")
    _git(local, "add", "tools/precommit.py")
    _git(local, "commit", "-m", "operator manual commit")

    # Working tree is clean; the auto-sync is invoked without a fresh
    # working-tree change. The valid ahead commit must still be pushed.
    assert _git_out(local, "status", "--porcelain") == ""
    assert sync_context_repo(
        local,
        include_paths=["tools/**"],
        exclude_paths=[],
        message="nanobot: sync test",
    )

    verify = tmp_path / "verify"
    _git(tmp_path, "clone", str(remote), str(verify))
    assert (verify / "tools" / "precommit.py").read_text(encoding="utf-8") == "# operator change\n"


# ---------------------------------------------------------------------------
# Test P: unselected dirty data survives byte-equivalent
# ---------------------------------------------------------------------------


def test_p_unselected_dirty_data_survives_byte_equivalent(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)

    (local / "tools" / "managed.py").write_text("managed\n", encoding="utf-8")
    # Excluded runtime data file with binary-ish content.
    excluded_dir = local / "agent-workspace" / "runs"
    excluded_dir.mkdir(parents=True)
    excluded_file = excluded_dir / "scratch.bin"
    excluded_payload = bytes(range(256)) * 4
    excluded_file.write_bytes(excluded_payload)

    before_mtime_ns = excluded_file.stat().st_mtime_ns

    assert sync_context_repo(
        local,
        include_paths=["tools/**"],
        exclude_paths=["agent-workspace/runs/**"],
        message="nanobot: sync test",
    )

    # The excluded file's content is byte-for-byte unchanged.
    assert excluded_file.read_bytes() == excluded_payload
    # And the file still exists (not deleted).
    assert excluded_file.exists()
    # mtime is preserved by `git pull --rebase --autostash` on the same
    # filesystem; we don't strictly require equality, only existence and
    # byte equivalence.
    assert excluded_file.stat().st_mtime_ns >= before_mtime_ns - 1


# ---------------------------------------------------------------------------
# Test Q: legitimate code commit under tools/ / modules/ / tests/ passes
# ---------------------------------------------------------------------------


def test_q_legitimate_code_commit_passes_path_policy(tmp_path: Path) -> None:
    remote, seed, local = _bootstrap_remote_with_initial(tmp_path)

    # Simulate a legitimate code/contract commit touching three project
    # roots: tools/, modules/, tests/. The include list permits all three.
    (local / "modules").mkdir(parents=True)
    (local / "tests").mkdir(parents=True)
    (local / "tools" / "feature.py").write_text("# new feature\n", encoding="utf-8")
    (local / "modules" / "feature.md").write_text("# module spec\n", encoding="utf-8")
    (local / "tests" / "test_feature.py").write_text("# tests\n", encoding="utf-8")
    _git(local, "add", "tools/feature.py", "modules/feature.md", "tests/test_feature.py")
    _git(local, "commit", "-m", "add feature")

    assert sync_context_repo(
        local,
        include_paths=["tools/**", "modules/**", "tests/**"],
        exclude_paths=[],
        message="nanobot: sync test",
    )

    verify = tmp_path / "verify"
    _git(tmp_path, "clone", str(remote), str(verify))
    assert (verify / "tools" / "feature.py").exists()
    assert (verify / "modules" / "feature.md").exists()
    assert (verify / "tests" / "test_feature.py").exists()
