import numpy as np
from scipy import optimize as root
from PySDM.simulation.physics import formulae


class _MinFun:
    def __init__(self, T, p, RH, kappa, r_d):
        self.T = T
        self.p = p
        self.RH = RH
        self.kappa = kappa
        self.r_d = r_d

    def __call__(self, r_w):
        return formulae.dr_dt_MM(r_w, self.T, self.p, self.RH, self.kappa, self.r_d)


def r_wet_init(r_dry: np.ndarray, ambient_air, cell_id: np.ndarray, kappa):
    r_wet = np.empty_like(r_dry)

    for i, r_d in enumerate(r_dry):
        cid = cell_id[i]
        # root-finding initial guess
        a = r_d
        b = formulae.r_cr(kappa, r_d, ambient_air['T'][cid])
        # minimisation
        f = _MinFun(
            ambient_air['T'][cid],
            ambient_air['p'][cid],
            np.minimum(1, ambient_air['RH'][cid]),
            kappa,
            r_d
        )
        r_wet[i] = root.brentq(f, a, b)

    return r_wet
