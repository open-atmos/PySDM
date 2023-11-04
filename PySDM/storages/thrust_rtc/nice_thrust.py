"""
a decorator triggering ThrustRTC.Wait() after each function call
"""
from PySDM.storages.thrust_rtc.conf import trtc


def nice_thrust(*, wait=False, debug_print=False):
    """
    a decorator triggering ThrustRTC.Wait() after each function call

    Parameters
    ----------
    wait : bool
        if True, wait for ThrustRTC to finish
    debug_print : bool
        if True, print function name
    """

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
