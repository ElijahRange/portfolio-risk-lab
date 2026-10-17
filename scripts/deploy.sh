#!/usr/bin/env bash
# (Re)start the dashboard container on port 8050. Pass --install after changing
# requirements.txt. Plain code edits go live on their own (gunicorn --reload).
#
# The container is the event's LinuxONE data-science image (numpy/pandas/scikit-learn
# prebuilt for s390x). Extra packages go in ./.venv, layered on top of the image's own
# /opt/venv. We don't `podman build` a custom image because podman 5.8 on this server
# fails to build on top of that image ("unsupported MIME type for compression").
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
NAME=dashboard
# keep-id runs as linux1 inside the container, so the app can write to the mounted repo.
COMMON=(--userns keep-id --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/app:z -w /app --entrypoint "")

[[ -f .env ]] || { echo "Missing .env - copy .env.example to .env and set DASH_USERS." >&2; exit 1; }

if [[ "${1:-}" == "--install" || ! -x .venv/bin/gunicorn ]]; then
  podman run --rm "${COMMON[@]}" "$BASE" bash -c '
    set -e
    [ -x .venv/bin/python ] || python -m venv .venv
    site=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
    printf "%s\n" /opt/venv/lib/python3.12/site-packages /opt/venv/lib64/python3.12/site-packages > "$site/base_image.pth"
    .venv/bin/pip install -q --no-cache-dir --disable-pip-version-check -r requirements.txt'
fi

mkdir -p data/cache
podman rm -f "$NAME" >/dev/null 2>&1 || true
podman run -d --name "$NAME" --network host --restart always "${COMMON[@]}" --env-file .env "$BASE" \
  .venv/bin/gunicorn app:server --bind 0.0.0.0:8050 --workers 2 --threads 4 --timeout 180 \
  --reload --reload-extra-file assets/style.css --access-logfile - >/dev/null
echo "Dashboard container started on port 8050 (logs: podman logs -f $NAME)"
