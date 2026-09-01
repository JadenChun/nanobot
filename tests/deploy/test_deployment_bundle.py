from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_nanobot_systemd_service_uses_context_repo_and_secrets() -> None:
    service = _read("deploy/systemd/nanobot-gateway.service")

    assert "User=marketing-agent" in service
    assert "WorkingDirectory=/opt/marketing-agent/client-marketing-assistance" in service
    assert "EnvironmentFile=/opt/marketing-agent/secrets/nanobot.env" in service
    assert "--config /opt/marketing-agent/.nanobot/config.json" in service
    assert "--log-file /opt/marketing-agent/logs/nanobot-gateway.log" in service
    assert (
        "ReadWritePaths=/opt/marketing-agent/.nanobot /opt/marketing-agent/client-marketing-assistance /opt/marketing-agent/logs /opt/marketing-agent/.cache /opt/marketing-agent/.codex /opt/marketing-agent/.local"
        in service
    )


def test_nanobot_config_example_attaches_marketing_context_repo() -> None:
    config = _read("deploy/nanobot-config.example.json")

    assert '"/opt/marketing-agent/client-marketing-assistance"' in config
    assert '"contextRepos"' in config
    assert '"autoSync": true' in config
    assert '"telegram"' in config
    assert '"enabled": true' in config
    assert '"agentBrowser"' in config
    assert '"enabled": false' in config
    assert '"fastCommands"' in config
    assert '"/connection_status"' in config


def test_nanobot_deploy_script_supports_checkout_as_install_directory() -> None:
    script = _read("deploy/scripts/deploy_nanobot_gateway.sh")

    assert 'SOURCE_REALPATH="$(realpath -m "${SOURCE_DIR}")"' in script
    assert 'INSTALL_REALPATH="$(realpath -m "${INSTALL_DIR}")"' in script
    assert 'if [[ "${SOURCE_REALPATH}" == "${INSTALL_REALPATH}" ]]; then' in script
    assert "skipping file copy" in script
    assert 'entry["autoSync"] = True' in script
    assert "Enabled context auto-sync" in script
    assert 'chmod 0640 "${CONFIG_FILE}" "${ENV_FILE}"' in script


def test_nanobot_update_script_skips_unchanged_running_service() -> None:
    script = _read("deploy/scripts/update_nanobot_gateway.sh")

    assert 'NANOBOT_BRANCH="${MARKETING_AGENT_NANOBOT_BRANCH:-main}"' in script
    assert 'before_head="$(run_as_deploy_user git -C "${NANOBOT_DIR}" rev-parse HEAD)"' in script
    assert 'after_head="$(run_as_deploy_user git -C "${NANOBOT_DIR}" rev-parse HEAD)"' in script
    assert "systemctl is-active --quiet nanobot-gateway" in script
    assert "skipping Nanobot redeploy" in script


def test_crawl4ai_worker_is_isolated_and_memory_bounded() -> None:
    service = _read("deploy/systemd/crawl4ai-worker.service")
    env_example = _read("deploy/crawl4ai-worker.env.example")

    assert "User=crawl4ai-worker" in service
    assert "Group=marketing-agent" in service
    assert "MemoryMax=1200M" in service
    assert "RuntimeDirectory=crawl4ai-worker" in service
    assert "IPAddressDeny=169.254.169.254/32" in service
    assert "ANTHROPIC" not in env_example
    assert "OPENAI" not in env_example
    assert "CRAWL4AI_ALLOWED_HOSTS=" in env_example


def test_nanobot_example_keeps_crawler_disabled_until_worker_is_ready() -> None:
    config = _read("deploy/nanobot-config.example.json")
    gateway_env = _read("deploy/nanobot.env.example")

    assert '"crawler"' in config
    assert '"maxToolIterations": 20' in config
    assert "CRAWL4AI_WORKER_ENABLED=false" in gateway_env


def test_local_marketing_launcher_uses_dedicated_authenticated_profile() -> None:
    launcher = _read("deploy/scripts/start_local_marketing_agent.ps1")
    setup = _read("deploy/scripts/setup_local_crawl4ai_profile.ps1")

    assert "--user-data-dir" in launcher
    assert '"--browser-channel", $browserChannel' in launcher
    assert '$browserChannel = "msedge"' in launcher
    assert "--headed" in launcher
    assert "egocat-marketing" in launcher
    assert 'CRAWL4AI_AUTH_PROFILE_ENABLED = "true"' in launcher
    assert "--no-logs" in launcher
    assert "--log-file" in launcher
    assert "facebook.com" in launcher
    assert "tiktok.com" in launcher
    assert "Microsoft\\Edge\\Application\\msedge.exe" in setup
    assert "Start-Process -FilePath $chrome" in setup
    assert ".nanobot-profile-ready" in setup
    assert "close every window" in setup
