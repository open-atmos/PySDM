""" Timescale derived for tritium with assumption of zero ambient concentration - see text above
    Table 1 [Bolin 1958](https://digitallibrary.un.org/record/3892725) """

import numpy as np

class Bolin1958:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.BOLIN_C1)

    @staticmethod
    # pylint: disable=too-many-arguments
    def tau(const, radius, r_dr_dt):
        return (-3 / radius**2 * r_dr_dt * const.BOLIN_C1)**-1
