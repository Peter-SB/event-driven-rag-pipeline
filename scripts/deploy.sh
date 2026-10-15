#!/usr/bin/env bash
# Deploy to the homelab: pull latest and rebuild the docker-compose stack.
# Assumes the repo is already cloned on the homelab at DEPLOY_PATH with its
# own .env in place. Run with bash (git bash on Windows works fine).
# Pushes the current branch first. Tip: `git deploy` (local alias) runs this.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

DEPLOY_USER="${DEPLOY_USER:?DEPLOY_USER not set — add it to .env}"
DEPLOY_HOST="${DEPLOY_HOST:?DEPLOY_HOST not set — add it to .env}"
DEPLOY_PATH="${DEPLOY_PATH:?DEPLOY_PATH not set — add it to .env}"
COMPOSE_FILE="${DEPLOY_COMPOSE_FILE:-docker-compose.yml}"

COMMIT_MSG="$(git log -1 --pretty=%s)"

echo "Starting deploy to ${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH} (${COMMIT_MSG})"

echo "Pushing local commits..."
git push

echo "Pulling latest..."
ssh "${DEPLOY_USER}@${DEPLOY_HOST}" "cd '${DEPLOY_PATH}' && git pull"

echo "Rebuilding and restarting stack (${COMPOSE_FILE})..."
ssh "${DEPLOY_USER}@${DEPLOY_HOST}" "cd '${DEPLOY_PATH}' && docker compose -f '${COMPOSE_FILE}' up -d --build"

echo ""
echo ""
echo "--------------------------------------"
echo "Finished deploy. Commit: ${COMMIT_MSG}"
echo "--------------------------------------"
