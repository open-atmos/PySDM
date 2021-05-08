import PySDM.physics.constants as const
from numpy import sqrt


class KappaKoehlerLeadingTerms:
    @staticmethod
    def RH_eq(r, T, kp, rd3, sgm):
        return (
           1 +
           (2 * sgm / const.Rv / T / const.rho_w) / r -
           kp * rd3 / r ** 3
        )

    @staticmethod
    def r_cr(kp, rd3, T, sgm):
        return sqrt(3 * kp * rd3 / (2 * sgm / const.Rv / T / const.rho_w))

