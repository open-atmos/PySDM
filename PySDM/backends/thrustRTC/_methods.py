"""
Created at 20.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .nice_thrust import nice_thrust
from .conf import NICE_THRUST_FLAGS


class Methods:

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def apply(function, args, output):
        raise NotImplementedError()

    # Warning (potentially inefficient): reduction
    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def first_element_is_zero(arr):
        # first_elem = arr.get(0)
        return arr.to_host()[0] == 0

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def _apply_f_3_3(function, arg0, arg1, arg2, output0, output1, output2):
        raise NotImplementedError()
