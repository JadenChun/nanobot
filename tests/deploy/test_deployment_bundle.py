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
    assert "ReadWritePaths=/opt/marketing-agent/.nanobot /opt/marketing-agent/client-marketing-assistance /opt/marketing-agent/logs /opt/marketing-agent/.cache /opt/marketing-agent/.codex /opt/marketing-agent/.local" in service


def test_nanobot_config_example_attaches_marketing_context_repo() -> None:
    config = _read("deploy/nanobot-config.example.json")

    assert '"/opt/marketing-agent/client-marketing-assistance"' in config
    assert '"contextRepos"' in config
    assert '"telegram"' in config
    assert '"enabled": true' in config
    assert '"fastCommands"' in config
    assert '"/connection_status"' in config


def test_nanobot_deploy_script_supports_checkout_as_install_directory() -> None:
    script = _read("deploy/scripts/deploy_nanobot_gateway.sh")

    assert 'SOURCE_REALPATH="$(realpath -m "${SOURCE_DIR}")"' in script
    assert 'INSTALL_REALPATH="$(realpath -m "${INSTALL_DIR}")"' in script
    assert 'if [[ "${SOURCE_REALPATH}" == "${INSTALL_REALPATH}" ]]; then' in script
    assert "skipping file copy" in script
    assert 'chmod 0640 "${CONFIG_FILE}" "${ENV_FILE}"' in script


def test_nanobot_update_script_skips_unchanged_running_service() -> None:
    script = _read("deploy/scripts/update_nanobot_gateway.sh")

    assert 'before_head="$(run_as_deploy_user git -C "${NANOBOT_DIR}" rev-parse HEAD)"' in script
    assert 'after_head="$(run_as_deploy_user git -C "${NANOBOT_DIR}" rev-parse HEAD)"' in script
    assert "systemctl is-active --quiet nanobot-gateway" in script
    assert "skipping Nanobot redeploy" in script
