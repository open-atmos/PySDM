"""
a decorator triggering ThrustRTC.Wait() after each function call
"""

from PySDM.backends.impl_thrust_rtc.conf import trtc


def nice_thrust(*, wait=False, debug_print=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if debug_print:
                print(func.__name__)
            result = func(*args, **kwargs)
            if wait:
                trtc.Wait()
            return result

        return wrapper

    return decorator
