"""
Created at 20.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc


class Methods:
    # Warning (potentially inefficient): reduction
    @staticmethod
    def first_element_is_zero(arr):
        return arr.get(0) == 0

    @staticmethod
    def apply_f_3_3(function, arg0, arg1, arg2, output0, output1, output2):
        raise NotImplementedError()

    @staticmethod
    def apply(function, args, output):
        raise NotImplementedError()
