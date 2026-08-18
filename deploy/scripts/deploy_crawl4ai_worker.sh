#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${MARKETING_AGENT_BASE_DIR:-/opt/marketing-agent}"
NANOBOT_DIR="${MARKETING_AGENT_NANOBOT_DIR:-${BASE_DIR}/nanobot-custom}"
WORKER_DIR="${MARKETING_AGENT_CRAWLER_DIR:-${BASE_DIR}/crawl4ai-worker}"
ENV_DIR="/etc/crawl4ai-worker"
ENV_FILE="${ENV_DIR}/crawl4ai-worker.env"
SERVICE_FILE="/etc/systemd/system/crawl4ai-worker.service"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root or with sudo." >&2
  exit 1
fi
if [[ ! -f "${NANOBOT_DIR}/pyproject.toml" ]]; then
  echo "Nanobot checkout is missing: ${NANOBOT_DIR}" >&2
  exit 1
fi
if ! getent group marketing-agent >/dev/null; then
  echo "Required group does not exist: marketing-agent" >&2
  exit 1
fi
if ! id crawl4ai-worker >/dev/null 2>&1; then
  useradd --system --home-dir /var/lib/crawl4ai-worker \
    --create-home --gid marketing-agent --shell /usr/sbin/nologin crawl4ai-worker
fi

install -d -m 0750 -o crawl4ai-worker -g marketing-agent "${WORKER_DIR}"
install -d -m 0750 -o root -g marketing-agent "${ENV_DIR}"

if [[ ! -x "${WORKER_DIR}/.venv/bin/python" ]]; then
  runuser -u crawl4ai-worker -- python3 -m venv "${WORKER_DIR}/.venv"
fi
runuser -u crawl4ai-worker -- "${WORKER_DIR}/.venv/bin/pip" install --upgrade pip
runuser -u crawl4ai-worker -- "${WORKER_DIR}/.venv/bin/pip" install \
  "${NANOBOT_DIR}[crawler-worker]"

# Crawl4AI's setup command installs the Chromium browser used by Playwright.
runuser -u crawl4ai-worker -- env \
  HOME=/var/lib/crawl4ai-worker \
  PLAYWRIGHT_BROWSERS_PATH=/var/lib/crawl4ai-worker/ms-playwright \
  "${WORKER_DIR}/.venv/bin/crawl4ai-setup"

if [[ ! -f "${ENV_FILE}" ]]; then
  install -m 0640 -o root -g marketing-agent \
    "${NANOBOT_DIR}/deploy/crawl4ai-worker.env.example" "${ENV_FILE}"
  echo "Created ${ENV_FILE}; review the host allowlist before enabling the worker."
fi

install -m 0644 "${NANOBOT_DIR}/deploy/systemd/crawl4ai-worker.service" "${SERVICE_FILE}"
systemctl daemon-reload
systemctl enable --now crawl4ai-worker
systemctl status crawl4ai-worker --no-pager

echo "Crawl4AI worker installed. Run the Unix-socket health check from deploy/README.md."
