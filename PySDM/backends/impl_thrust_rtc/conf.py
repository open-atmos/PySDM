from PySDM.backends.impl_thrust_rtc.test_helpers._flag import fakeThrustRTC

if not fakeThrustRTC:
    # noinspection PyUnresolvedReferences
    import ThrustRTC as trtc  # pylint: disable=unused-import
    # noinspection PyUnresolvedReferences
    import CURandRTC as rndrtc  # pylint: disable=unused-import
else:
    # noinspection PyUnresolvedReferences
    from .test_helpers.fake_thrust_rtc import FakeThrustRTC as trtc  # pylint: disable=unused-import
    # noinspection PyUnresolvedReferences
    rndrtc = None

NICE_THRUST_FLAGS = dict(
    wait=False,
    debug_print=False
)
