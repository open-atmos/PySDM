"""
kappa-Koehler parameterization
 ([Petters & Kreidenweis 2007](https://doi.org/10.5194/acp-7-1961-2007))
"""

import numpy as np


class KappaKoehler:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def RH_eq(const, r, T, kp, rd3, sgm):
        return (
            np.exp((2 * sgm / const.Rv / T / const.rho_w) / r)
            * (r**3 - rd3)
            / (r**3 - rd3 * (1 - kp))
        )

    @staticmethod
    def r_cr(const, kp, rd3, T, sgm):
        # TODO #493
        return np.sqrt(3 * kp * rd3 / (2 * sgm / const.Rv / T / const.rho_w))
