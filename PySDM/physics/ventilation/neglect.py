"""
constant ventilation coefficient of unity (i.e., neglect ventilation effects)
"""

import numpy as np


class Neglect:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def ventilation_coefficient(sqrt_re_times_cbrt_sc):
        return np.power(sqrt_re_times_cbrt_sc, 0)
