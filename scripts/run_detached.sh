#!/usr/bin/env bash
# Run a command (deploy.sh) in its own transient systemd scope, so the container's conmon process outlives the caller.
#
# Without this, a container started from the one-shot dashboard-sync.service lives in that service's cgroup, and
# systemd kills its conmon when the service finishes. The container keeps serving, but every write to stdout/stderr
# then fails with "Broken pipe" (the 2026-10-18 chat outage).
set -euo pipefail
if command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  exec systemd-run --user --scope --quiet --collect --unit="prism-$(basename "$1" .sh)-$(date +%s)" "$@"
fi
exec "$@"
