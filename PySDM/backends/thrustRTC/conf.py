"""
Created at 20.05.2020
"""

from PySDM.backends.thrustRTC.fakeThrustRTC._flag import fakeThrustRTC

# noinspection PyUnresolvedReferences
import CURandRTC as rndrtc
if not fakeThrustRTC:
    # noinspection PyUnresolvedReferences
    import ThrustRTC as trtc
else:
    # noinspection PyUnresolvedReferences
    from .fakeThrustRTC.fakeThrustRTC import FakeThrustRTC as trtc

NICE_THRUST_FLAGS = dict(
    wait=False,
    debug_print=False
)
