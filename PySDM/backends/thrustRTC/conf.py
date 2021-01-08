"""
Created at 20.05.2020
"""

from PySDM.backends.thrustRTC.fakeThrustRTC._flag import fakeThrustRTC
import os
import warnings

allowFakeThrustRTC = 'CI' in os.environ

if not fakeThrustRTC:
    # noinspection PyUnresolvedReferences
    import ThrustRTC as trtc
    # noinspection PyUnresolvedReferences
    import CURandRTC as rndrtc
elif allowFakeThrustRTC:
    # noinspection PyUnresolvedReferences
    from .fakeThrustRTC.fakeThrustRTC import FakeThrustRTC as trtc
    if 'CI' not in os.environ:
        warnings.warn('using FakeThrustRTC')
    # noinspection PyUnresolvedReferences
    rndrtc = None
else:
    raise ImportError("ThrustRTC is not available and FakeThrustRTC is not allowed. \n"
                      f"You can set 'allowFakeThrustRTC = True' in thrustRTC/conf.py")

NICE_THRUST_FLAGS = dict(
    wait=False,
    debug_print=False
)
