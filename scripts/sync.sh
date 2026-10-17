#!/usr/bin/env bash
# Pull the latest main from GitHub onto the server (run every minute by a systemd timer).
# Gunicorn --reload picks up code changes; dependency changes trigger an image rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."

git remote get-url origin >/dev/null 2>&1 || exit 0
before=$(git rev-parse HEAD)
git fetch -q origin main
if ! git merge -q --ff-only origin/main; then
  echo "Server copy has diverged from GitHub; not updating. Commit/push or reset the server copy." >&2
  exit 0
fi
after=$(git rev-parse HEAD)
if [[ "$before" != "$after" ]] && git diff --name-only "$before" "$after" | grep -qx requirements.txt; then
  scripts/deploy.sh --install
fi
