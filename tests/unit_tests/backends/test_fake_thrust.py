from PySDM.backends.impl_thrust_rtc.test_helpers.fake_thrust_rtc import FakeThrustRTC

def test_device_vector_fails_on_zero_size():
    # arrange
    sut = FakeThrustRTC.device_vector

    # act
    exception = None
    try:
        sut('float', size=0)
    except Exception as caught:
        exception = caught

    # assert
    assert exception is not None
