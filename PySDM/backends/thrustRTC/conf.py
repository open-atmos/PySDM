"""
Created at 20.05.2020
"""

from PySDM.backends.thrustRTC.fakeThrustRTC._flag import fakeThrustRTC

if not fakeThrustRTC:
    # noinspection PyUnresolvedReferences
    import ThrustRTC as trtc
    # noinspection PyUnresolvedReferences
    import CURandRTC as rndrtc
else:
    # noinspection PyUnresolvedReferences
    from .fakeThrustRTC.fakeThrustRTC import FakeThrustRTC as trtc
    # noinspection PyUnresolvedReferences
    rndrtc = None

NICE_THRUST_FLAGS = dict(
    wait=False,
    debug_print=False
)
