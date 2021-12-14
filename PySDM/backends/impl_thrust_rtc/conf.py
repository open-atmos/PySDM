"""
ThrustRTC import logic implementing the switch between the genuine ThrustRTC and FakeThrustRTC;
default nice_thrust flags
"""
from warnings import warn
from PySDM.backends.impl_thrust_rtc.test_helpers.flag import fakeThrustRTC

if not fakeThrustRTC:
    try:
        # noinspection PyUnresolvedReferences
        import ThrustRTC as trtc  # pylint: disable=unused-import
    except OSError as error:
        warn(f"importing ThrustRTC failed with {error}")
    try:
        # noinspection PyUnresolvedReferences
        import CURandRTC as rndrtc  # pylint: disable=unused-import
    except OSError as error:
        warn(f"importing CURandRTC failed with {error}")
else:
    # noinspection PyUnresolvedReferences
    from .test_helpers.fake_thrust_rtc import FakeThrustRTC as trtc  # pylint: disable=unused-import
    # noinspection PyUnresolvedReferences
    rndrtc = None

NICE_THRUST_FLAGS = dict(
    wait=False,
    debug_print=False
)
