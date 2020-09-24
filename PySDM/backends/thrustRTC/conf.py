"""
Created at 20.05.2020
"""

from ._flag import fakeThrustRTC

if not fakeThrustRTC:
    import ThrustRTC as trtc
    import CURandRTC as rndrtc
else:
    from .fake_thrust import FakeThrustRTC as trtc
    from .fake_thrust import FakeRandRTC as rndrtc

NICE_THRUST_FLAGS = dict(
    wait=False,
    debug_print=False
)
