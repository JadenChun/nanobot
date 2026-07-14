#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${MARKETING_AGENT_BASE_DIR:-/opt/marketing-agent}"
DEPLOY_USER="${MARKETING_AGENT_DEPLOY_USER:-marketing-agent}"
NANOBOT_DIR="${MARKETING_AGENT_NANOBOT_DIR:-${BASE_DIR}/nanobot-custom}"
NANOBOT_BRANCH="${MARKETING_AGENT_NANOBOT_BRANCH:-main}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root or with sudo." >&2
  exit 1
fi
if ! id "${DEPLOY_USER}" >/dev/null 2>&1; then
  echo "Deploy user does not exist: ${DEPLOY_USER}" >&2
  exit 1
fi
if [[ ! -d "${NANOBOT_DIR}/.git" ]]; then
  echo "Nanobot repo is missing: ${NANOBOT_DIR}" >&2
  exit 1
fi

run_as_deploy_user() {
  runuser -u "${DEPLOY_USER}" -- "$@"
}

before_head="$(run_as_deploy_user git -C "${NANOBOT_DIR}" rev-parse HEAD)"
echo "Updating Nanobot (${NANOBOT_BRANCH})..."
run_as_deploy_user git -C "${NANOBOT_DIR}" fetch origin
if run_as_deploy_user git -C "${NANOBOT_DIR}" rev-parse --verify --quiet "${NANOBOT_BRANCH}" >/dev/null; then
  run_as_deploy_user git -C "${NANOBOT_DIR}" switch "${NANOBOT_BRANCH}"
else
  run_as_deploy_user git -C "${NANOBOT_DIR}" switch -c "${NANOBOT_BRANCH}" --track "origin/${NANOBOT_BRANCH}"
fi
run_as_deploy_user git -C "${NANOBOT_DIR}" pull --ff-only origin "${NANOBOT_BRANCH}"
after_head="$(run_as_deploy_user git -C "${NANOBOT_DIR}" rev-parse HEAD)"

if [[ "${before_head}" != "${after_head}" ]]; then
  echo "Nanobot updated: ${before_head} -> ${after_head}; deploying gateway service..."
  bash "${NANOBOT_DIR}/deploy/scripts/deploy_nanobot_gateway.sh" "${NANOBOT_DIR}"
elif ! systemctl is-active --quiet nanobot-gateway; then
  echo "Nanobot code unchanged, but service is not active; deploying gateway service..."
  bash "${NANOBOT_DIR}/deploy/scripts/deploy_nanobot_gateway.sh" "${NANOBOT_DIR}"
else
  echo "Nanobot code unchanged and service is active; skipping Nanobot redeploy."
fi

systemctl status nanobot-gateway --no-pager || true
