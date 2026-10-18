"""Keep the app working if its stdout/stderr pipes break. Imported first by app.py.

Inside the podman container, everything the app prints goes through the container's `conmon` process. On 2026-10-18
conmon was killed (it belonged to the systemd job that had restarted the container), so every write to
stdout/stderr raised BrokenPipeError. Loading the chat model prints a progress bar, so the assistant failed on every
question until the container was restarted. These wrappers drop output that can't be written instead of raising.
scripts/watchdog.sh also restarts the container when conmon is gone.
"""
import sys


class _SafeStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        try:
            return self._stream.write(data)
        except (OSError, ValueError):  # broken pipe, or the stream was closed
            return len(data)

    def flush(self):
        try:
            self._stream.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is not None and not isinstance(getattr(sys, _name), _SafeStream):
        setattr(sys, _name, _SafeStream(getattr(sys, _name)))
