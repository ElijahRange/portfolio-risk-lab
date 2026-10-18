#!/usr/bin/env bash
# Restart the dashboard container when it is broken. sync.sh runs this every minute (dashboard-sync.timer).
#
# Restarts when: the container is missing or stopped; its conmon process is gone (then every write to
# stdout/stderr fails with "Broken pipe": the 2026-10-18 outage); or GET /healthz fails 3 minutes in a row
# (the portfolio doesn't load or the on-device assistant can't answer). At most one restart per 15 minutes.
# Log: ~/.cache/prism-watchdog/log. Test without restarting: DRY_RUN=1 scripts/watchdog.sh
set -uo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-dashboard}
PORT=${PORT:-8050}
STATE=${STATE:-$HOME/.cache/prism-watchdog}
mkdir -p "$STATE"
log() { echo "$(date -Is) $*" | tee -a "$STATE/log"; }

problem=""
if ! podman container exists "$NAME"; then
  problem="container $NAME is missing"
elif [[ "$(podman inspect -f '{{.State.Running}}' "$NAME")" != "true" ]]; then
  problem="container $NAME is not running"
else
  conmon=$(podman inspect -f '{{.State.ConmonPid}}' "$NAME")
  if [[ -z "$conmon" || "$conmon" == "0" ]] || ! kill -0 "$conmon" 2>/dev/null; then
    problem="conmon ($conmon) is gone, so the app's output pipes are broken"
  elif curl -fsS -m 30 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    echo 0 > "$STATE/fails"
  else
    fails=$(( $(cat "$STATE/fails" 2>/dev/null || echo 0) + 1 ))
    echo "$fails" > "$STATE/fails"
    (( fails >= 3 )) && problem="/healthz failed $fails minutes in a row"
  fi
fi
[[ -z "$problem" ]] && exit 0

now=$(date +%s)
last=$(cat "$STATE/last_restart" 2>/dev/null || echo 0)
if (( now - last < 900 )); then
  log "$problem; restarted less than 15 minutes ago, so waiting"
  exit 0
fi
log "$problem; restarting $NAME"
[[ -n "${DRY_RUN:-}" ]] && exit 0
echo "$now" > "$STATE/last_restart"
echo 0 > "$STATE/fails"
scripts/run_detached.sh scripts/deploy.sh
