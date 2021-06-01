from PySDM.backends.thrustRTC.test_helpers._flag import fakeThrustRTC

if not fakeThrustRTC:
    # noinspection PyUnresolvedReferences
    import ThrustRTC as trtc
    # noinspection PyUnresolvedReferences
    import CURandRTC as rndrtc
else:
    # noinspection PyUnresolvedReferences
    from .test_helpers.fakeThrustRTC import FakeThrustRTC as trtc
    # noinspection PyUnresolvedReferences
    rndrtc = None

NICE_THRUST_FLAGS = dict(
    wait=False,
    debug_print=False
)
