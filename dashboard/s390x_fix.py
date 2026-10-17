"""Make Plotly charts correct on IBM Z (s390x), which is a big-endian machine.

Plotly (6+) sends numeric arrays to the browser as base64 "typed arrays" using the
server's native byte order, but plotly.js always decodes them as little-endian. On
s390x that turns every number into garbage (e.g. +20% renders as 7e+276%). This
re-encodes the bytes as little-endian. On little-endian machines it does nothing.
"""
import base64
import sys

import numpy as np
import _plotly_utils.utils as plotly_utils

if sys.byteorder == "big" and not getattr(plotly_utils, "_s390x_patched", False):
    _original = plotly_utils.to_typed_array_spec

    def to_typed_array_spec(v):
        spec = _original(v)
        if isinstance(spec, dict) and "bdata" in spec:
            native = np.frombuffer(base64.b64decode(spec["bdata"]), dtype=np.dtype(spec["dtype"]))
            little = native.astype(native.dtype.newbyteorder("<"))
            spec["bdata"] = base64.b64encode(little.tobytes()).decode("ascii")
        return spec

    # convert_to_base64() looks this name up at call time, so patching the module is enough.
    plotly_utils.to_typed_array_spec = to_typed_array_spec
    plotly_utils._s390x_patched = True
