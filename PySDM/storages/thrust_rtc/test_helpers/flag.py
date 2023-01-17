"""
flag enabling `PySDM.backends.impl_thrust_rtc.test_helpers.fake_thrust_rtc.FakeThrustRTC`
 (for tests of GPU code on machines with no GPU)
"""
import os

fakeThrustRTC = os.getenv("FAKE_THRUST_RTC", "0") == "1"
