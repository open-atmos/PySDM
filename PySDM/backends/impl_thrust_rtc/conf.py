"""
ThrustRTC import logic (the backend is disabled if ThrustRTC cannot be imported, e.g., if no
CUDA driver is present); default nice_thrust flags
"""

import os
from warnings import warn

ENABLE = True
try:
    # noinspection PyUnresolvedReferences
    import ThrustRTC as trtc  # pylint: disable=unused-import

    if "NVRTC_PATH" in os.environ:
        trtc.set_libnvrtc_path(os.environ["NVRTC_PATH"])
except OSError as error:
    warn(f"importing ThrustRTC failed with {error}")
except ImportError:  # raised by ThrustRTC if CUDA initialisation fails
    ENABLE = False
try:
    # noinspection PyUnresolvedReferences
    import CURandRTC as rndrtc  # pylint: disable=unused-import
except OSError as error:
    warn(f"importing CURandRTC failed with {error}")
except ImportError:
    ENABLE = False


class _Unavailable:
    """placeholder for the ThrustRTC and CURandRTC modules, which allows
    the backend modules to be imported (e.g., by pdoc) without CUDA"""

    def __getattr__(self, _):
        return self

    def __call__(self, *_, **__):
        return self


if not ENABLE:
    trtc = rndrtc = _Unavailable()

NICE_THRUST_FLAGS = {
    "wait": False,
    "debug_print": False,
}  # TODO #1120: move to GPU backend ctor
