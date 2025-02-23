"""
ThrustRTC import logic implementing the switch between the genuine ThrustRTC and FakeThrustRTC;
default nice_thrust flags
"""

import os
from warnings import warn

from PySDM.backends.impl_thrust_rtc.test_helpers.flag import fakeThrustRTC

if not fakeThrustRTC:
    try:
        # noinspection PyUnresolvedReferences
        import ThrustRTC as trtc  # pylint: disable=unused-import

        if "NVRTC_PATH" in os.environ:
            trtc.set_libnvrtc_path(os.environ["NVRTC_PATH"])
    except OSError as error:
        warn(f"importing ThrustRTC failed with {error}")
    try:
        # noinspection PyUnresolvedReferences
        import CURandRTC as rndrtc  # pylint: disable=unused-import
    except OSError as error:
        warn(f"importing CURandRTC failed with {error}")
else:
    # noinspection PyUnresolvedReferences
    # fmt: off
    # isort: off
    from .test_helpers.fake_thrust_rtc import (  # pylint: disable=unused-import
        FakeThrustRTC as trtc,
    )
    # isort: on
    # fmt: on
    # noinspection PyUnresolvedReferences
    rndrtc = None

NICE_THRUST_FLAGS = {
    "wait": False,
    "debug_print": False,
}  # TODO #1120: move to GPU backend ctor
