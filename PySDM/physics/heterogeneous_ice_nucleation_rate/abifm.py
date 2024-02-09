"""
ABIFM heterogeneous freezing rate parameterization
 ([Knopf & Alpert 2013](https://doi.org/10.1039/C3FD00035D))
"""

import numpy as np


class ABIFM:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.ABIFM_M)
        assert np.isfinite(const.ABIFM_C)

    @staticmethod
    def j_het(const, a_w_ice):
        return 10 ** (const.ABIFM_M * (1 - a_w_ice) + const.ABIFM_C) * const.ABIFM_UNIT
