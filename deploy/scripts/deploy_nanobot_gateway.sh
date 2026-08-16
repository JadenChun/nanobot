#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-$(pwd)}"
INSTALL_DIR="${MARKETING_AGENT_NANOBOT_DIR:-/opt/marketing-agent/nanobot-custom}"
CONFIG_FILE="${MARKETING_AGENT_NANOBOT_CONFIG:-/opt/marketing-agent/.nanobot/config.json}"
ENV_FILE="${MARKETING_AGENT_NANOBOT_ENV:-/opt/marketing-agent/secrets/nanobot.env}"
CONTEXT_REPO="${MARKETING_AGENT_CONTEXT_REPO:-/opt/marketing-agent/client-marketing-assistance}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root or with sudo." >&2
  exit 1
fi
if [[ ! -f "${SOURCE_DIR}/pyproject.toml" || ! -f "${SOURCE_DIR}/nanobot/cli/commands.py" ]]; then
  echo "Source directory is not a nanobot checkout: ${SOURCE_DIR}" >&2
  exit 1
fi
if ! id marketing-agent >/dev/null 2>&1; then
  echo "Create the marketing-agent deploy user first." >&2
  exit 1
fi

SOURCE_REALPATH="$(realpath -m "${SOURCE_DIR}")"
INSTALL_REALPATH="$(realpath -m "${INSTALL_DIR}")"

install -d -m 0750 -o marketing-agent -g marketing-agent "${INSTALL_DIR}"
install -d -m 0750 -o marketing-agent -g marketing-agent /opt/marketing-agent/.nanobot
install -d -m 0750 -o marketing-agent -g marketing-agent /opt/marketing-agent/.nanobot/workspace
install -d -m 0750 -o marketing-agent -g marketing-agent /opt/marketing-agent/.cache
install -d -m 0750 -o marketing-agent -g marketing-agent /opt/marketing-agent/.codex
install -d -m 0750 -o marketing-agent -g marketing-agent /opt/marketing-agent/.local
install -d -m 0750 -o marketing-agent -g marketing-agent /opt/marketing-agent/logs
install -d -m 0750 -o root -g marketing-agent /opt/marketing-agent/secrets
touch /opt/marketing-agent/logs/nanobot-gateway.log
chown marketing-agent:marketing-agent /opt/marketing-agent/logs/nanobot-gateway.log
chmod 0640 /opt/marketing-agent/logs/nanobot-gateway.log

if [[ "${SOURCE_REALPATH}" == "${INSTALL_REALPATH}" ]]; then
  echo "Source directory is the install directory; skipping file copy."
else
  rsync -a --delete \
    --exclude '.git/' \
    --exclude '.mypy_cache/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    "${SOURCE_DIR}/" "${INSTALL_DIR}/"

  chown -R marketing-agent:marketing-agent "${INSTALL_DIR}"
fi

if [[ ! -x "${INSTALL_DIR}/.venv/bin/python" ]]; then
  runuser -u marketing-agent -- python3 -m venv "${INSTALL_DIR}/.venv"
fi
runuser -u marketing-agent -- "${INSTALL_DIR}/.venv/bin/pip" install --upgrade pip
runuser -u marketing-agent -- "${INSTALL_DIR}/.venv/bin/pip" install "${INSTALL_DIR}"
runuser -u marketing-agent -- "${INSTALL_DIR}/.venv/bin/python" -m compileall -q "${INSTALL_DIR}/nanobot"

install -m 0644 "${INSTALL_DIR}/deploy/systemd/nanobot-gateway.service" \
  /etc/systemd/system/nanobot-gateway.service
systemctl daemon-reload

if [[ ! -f "${CONFIG_FILE}" ]]; then
  install -m 0640 -o root -g marketing-agent \
    "${INSTALL_DIR}/deploy/nanobot-config.example.json" "${CONFIG_FILE}"
  echo "Created ${CONFIG_FILE}. Fill Telegram and provider secrets before starting Nanobot."
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  install -m 0640 -o root -g marketing-agent \
    "${INSTALL_DIR}/deploy/nanobot.env.example" "${ENV_FILE}"
  echo "Created ${ENV_FILE}. Fill OAUTH_BROKER_INTERNAL_API_KEY before starting Nanobot."
fi

# Existing installs keep their config across deploys. Migrate the marketing
# context entry so generated Markdown memory is committed and pushed again.
python3 - "${CONFIG_FILE}" "${CONTEXT_REPO}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

config_path = Path(sys.argv[1])
context_repo = str(Path(sys.argv[2]))
payload = json.loads(config_path.read_text(encoding="utf-8"))
entries = payload.get("agents", {}).get("defaults", {}).get("contextRepos", [])
changed = False
for entry in entries:
    if not isinstance(entry, dict):
        continue
    if str(Path(str(entry.get("path", "")))) != context_repo:
        continue
    if entry.get("autoSync") is not True:
        entry["autoSync"] = True
        changed = True

if changed:
    fd, temporary = tempfile.mkstemp(prefix=".nanobot-config-", suffix=".json", dir=config_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Enabled context auto-sync in {config_path}")
PY

chown root:marketing-agent "${CONFIG_FILE}" "${ENV_FILE}"
chmod 0640 "${CONFIG_FILE}" "${ENV_FILE}"

if grep -Eq '^OAUTH_BROKER_INTERNAL_API_KEY=.+$' "${ENV_FILE}" \
  && grep -Eq '"token"[[:space:]]*:[[:space:]]*"[^"]+"' "${CONFIG_FILE}"; then
  systemctl enable nanobot-gateway
  systemctl restart nanobot-gateway
  echo "Nanobot gateway deployed and restart requested."
else
  echo "Nanobot gateway files installed. Fill ${CONFIG_FILE} and ${ENV_FILE}, then run:"
  echo "  sudo systemctl enable --now nanobot-gateway"
fi
