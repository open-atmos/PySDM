import PySDM.physics.constants as const
from numpy import sqrt, exp


class KappaKoehler:
    @staticmethod
    def RH_eq(r, T, kp, rd3, sgm):
        return exp((2 * sgm / const.Rv / T / const.rho_w) / r) * (r**3 - rd3) / (r**3 - rd3 * (1-kp))

    @staticmethod
    def r_cr(kp, rd3, T, sgm):
        # TODO #493
        return sqrt(3 * kp * rd3 / (2 * sgm / const.Rv / T / const.rho_w))
