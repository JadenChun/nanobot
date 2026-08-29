from __future__ import annotations

import subprocess
from pathlib import Path

from nanobot.utils import git_sync
from nanobot.utils.git_sync import _select_changed_paths, sync_context_repo


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


def test_select_changed_paths_filters_include_and_exclude(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _configure_user(repo)
    (repo / "outputs").mkdir()
    (repo / "config").mkdir()
    (repo / "data").mkdir()
    (repo / "outputs" / "draft.md").write_text("draft", encoding="utf-8")
    (repo / "config" / "secret.json").write_text("secret", encoding="utf-8")
    (repo / "data" / "seo-pipeline.json").write_text("{}", encoding="utf-8")

    selected = _select_changed_paths(
        repo,
        include_paths=["outputs/**", "data/seo-pipeline.json"],
        exclude_paths=["config/**", "**/*secret*.json"],
    )

    assert sorted(selected) == ["data/seo-pipeline.json", "outputs/draft.md"]


def test_sync_context_repo_recovers_pull_conflict_by_reapplying_selected_changes(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    local = tmp_path / "local"
    other = tmp_path / "other"

    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure_user(seed)
    (seed / "agent-workspace" / "outputs").mkdir(parents=True)
    (seed / "agent-workspace" / "outputs" / "draft.md").write_text("base\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "branch", "-M", "main")
    _git(seed, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")

    _git(tmp_path, "clone", str(remote), str(local))
    _git(tmp_path, "clone", str(remote), str(other))
    _configure_user(local)
    _configure_user(other)

    draft = Path("agent-workspace/outputs/draft.md")
    (other / draft).write_text("remote change\n", encoding="utf-8")
    _git(other, "commit", "-am", "remote update")
    _git(other, "push")

    (local / draft).write_text("local managed change\n", encoding="utf-8")

    assert sync_context_repo(
        local,
        include_paths=["agent-workspace/outputs/**"],
        exclude_paths=[],
        message="nanobot: sync test",
    )

    assert (local / draft).read_text(encoding="utf-8") == "local managed change\n"
    assert _git_out(local, "status", "--porcelain") == ""

    verify = tmp_path / "verify"
    _git(tmp_path, "clone", str(remote), str(verify))
    assert (verify / draft).read_text(encoding="utf-8") == "local managed change\n"


def test_sync_context_repo_recovers_push_rejection_conflict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    local = tmp_path / "local"
    other = tmp_path / "other"

    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure_user(seed)
    (seed / "agent-workspace" / "outputs").mkdir(parents=True)
    (seed / "agent-workspace" / "outputs" / "draft.md").write_text("base\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "branch", "-M", "main")
    _git(seed, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")

    _git(tmp_path, "clone", str(remote), str(local))
    _git(tmp_path, "clone", str(remote), str(other))
    _configure_user(local)
    _configure_user(other)

    draft = Path("agent-workspace/outputs/draft.md")
    (local / draft).write_text("local managed change\n", encoding="utf-8")

    original_run_git = git_sync._run_git
    pushed_remote_update = False

    def run_git_with_remote_race(repo: Path, *args: str, timeout: int = 30):
        nonlocal pushed_remote_update
        if repo == local and args and args[0] == "push" and not pushed_remote_update:
            (other / draft).write_text("remote race change\n", encoding="utf-8")
            _git(other, "commit", "-am", "remote race update")
            _git(other, "push")
            pushed_remote_update = True
        return original_run_git(repo, *args, timeout=timeout)

    monkeypatch.setattr(git_sync, "_run_git", run_git_with_remote_race)

    assert sync_context_repo(
        local,
        include_paths=["agent-workspace/outputs/**"],
        exclude_paths=[],
        message="nanobot: sync test",
    )

    assert pushed_remote_update is True
    assert (local / draft).read_text(encoding="utf-8") == "local managed change\n"
    # The new contract intentionally leaves unselected dirty files
    # untouched in the working tree. We assert instead that the
    # selected change was pushed to the remote after recovery.
    assert _git_out(local, "log", "origin/main..HEAD", "--oneline") == ""

    verify = tmp_path / "verify-push-rejection"
    _git(tmp_path, "clone", str(remote), str(verify))
    assert (verify / draft).read_text(encoding="utf-8") == "local managed change\n"
