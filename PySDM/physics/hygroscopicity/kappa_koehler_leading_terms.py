"""
leading-terms of the kappa-Koehler parameterization resulting in classic
 two-term formulation with 1/r and 1/r^3 terms corresponding to surface-tension
 (Kelvin) and soluble substance (Raoult/Wüllner) effects, respectively
"""

import numpy as np


class KappaKoehlerLeadingTerms:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def RH_eq(const, r, T, kp, rd3, sgm):
        return (
            1
            + (2 * sgm / const.Rv / T / const.rho_w) / r
            - kp * rd3 / np.power(r, const.THREE)
        )

    @staticmethod
    def r_cr(const, kp, rd3, T, sgm):
        return np.sqrt(3 * kp * rd3 / (2 * sgm / const.Rv / T / const.rho_w))
