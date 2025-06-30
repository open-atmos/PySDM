"""Timescale derived for tritium with assumption of zero ambient concentration - see text above
Table 1 [Bolin 1958](https://digitallibrary.un.org/record/3892725)"""

import numpy as np


class Bolin1958:  # pylint: disable=too-few-public-methods
    """Implementation of timescale used by Bolin 1958"""

    def __init__(self, const):
        assert np.isfinite(const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1)

    @staticmethod
    def tau(const, dm_dt_over_m):
        return 1 / (const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1 * dm_dt_over_m)
