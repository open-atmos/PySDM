"""Bolin number"""

import numpy as np


class Bolin1958:  # pylint: disable=too-few-public-methods
    """Implementation of timescale used by Bolin 1958"""

    def __init__(self, const):
        assert np.isfinite(const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1)

    @staticmethod
    def bolin_number(const):
        return const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1
