"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State
from PySDM.simulation import phys
from PySDM.simulation.phys import si
from PySDM.simulation.ambient_air.moist_air import MoistAir
from PySDM.utils import Physics

import numpy as np
from scipy import optimize as root


class Condensation:
    def __init__(self, ambient_air: MoistAir, dt, kappa):
        self.ambient_air = ambient_air

        self.dt = dt * si.seconds
        self.kappa = kappa

        self.rd = None

    @staticmethod
    def r_wet_init(r_dry: np.ndarray, ambient_air: MoistAir, cell_id: np.ndarray, kappa):
        r_wet = np.empty_like(r_dry)

        for i, r_d in enumerate(r_dry):
            cid = cell_id[i]
            # root-finding initial guess
            a = r_d
            b = phys.mgn(Condensation.r_cr(kappa, r_d * si.metres, ambient_air.T[cid] * si.kelvins))
            # minimisation
            f = lambda r_w: phys.mgn(Condensation.dr_dt_MM(
                r_w * si.metres,
                ambient_air.T[cid] * si.kelvin,
                ambient_air.p[cid] * si.pascal,
                np.minimum(0, ambient_air.RH[cid] - 1) * si.dimensionless,
                kappa,
                r_d * si.metres
            ))
            r_wet[i] = root.brentq(f, a, b)

        return r_wet

    def __call__(self, state: State):
        self.ambient_air.sync()

        x = state.get_backend_storage("x")
        rdry = state.get_backend_storage("dry radius")

        for i in state.idx[:state.SD_num]:
            cid = state.cell_id[i]
            r = Physics.x2r(x[i]) * si.metres
            T = self.ambient_air.T[cid] * si.kelvin
            p = self.ambient_air.p[cid] * si.pascal
            S = (self.ambient_air.RH[cid] - 1) * si.dimensionless
            kp = self.kappa
            rd = rdry[i] * si.metres

            # explicit Euler
            r_new = r + self.dt * self.dr_dt_MM(r, T, p, S, kp, rd)

            x[i] = Physics.r2x(phys.mgn(r_new))

        # TODO: update drop radii
        #       update fields due to condensation/evaporation
        #       ensure the above does include droplets that precipitated out of the domain

    @staticmethod
    def dr_dt_MM(r, T, p, S, kp, rd):
        nom = (S - Condensation.A(T) / r + Condensation.B(kp, rd) / r ** 3)
        den = Condensation.Fd(T, Condensation.D(r, T)) + Condensation.Fk(T, Condensation.K(r, T, p), Condensation.lv(T))
        result = 1 / r * nom / den
        return result

    @staticmethod
    def lv(T):
        # latent heat of evaporation
        return phys.l_tri + (phys.c_pv - phys.c_pw) * (T - phys.T_tri)

    @staticmethod
    def lambdaD(T):
        return phys.D0 / np.sqrt(2 * phys.Rv * T)

    @staticmethod
    def lambdaK(T, p):
        return (4 / 5) * phys.K0 * T / p / np.sqrt(2 * phys.Rd * T)

    @staticmethod
    def beta(Kn):
        return (1 + Kn) / (1 + 1.71 * Kn + 1.33 * Kn * Kn)

    @staticmethod
    def D(r, T):
        Kn = Condensation.lambdaD(T) / r  # TODO: optional
        return phys.D0 * Condensation.beta(Kn)

    @staticmethod
    def K(r, T, p):
        Kn = Condensation.lambdaK(T, p) / r
        return phys.K0 * Condensation.beta(Kn)

    # Maxwel-Mason coefficients:
    @staticmethod
    def Fd(T, D):
        return phys.rho_w * phys.Rv * T / D / Condensation.pvs(T)

    @staticmethod
    def Fk(T, K, lv):
        return phys.rho_w * lv / K / T * (lv / phys.Rv / T - 1)

    # Koehler curve (expressed in partial pressure):
    @staticmethod
    def A(T):
        return 2 * phys.sgm / phys.Rv / T / phys.rho_w

    @staticmethod
    def B(kp, rd):
        return kp * rd ** 3

    @staticmethod
    def r_cr(kp, rd, T):
        # critical radius
        return np.sqrt(3 * kp * rd ** 3 / Condensation.A(T))

    @staticmethod
    def pvs(T):
        # August-Roche-Magnus formula
        return phys.ARM_C1 * np.exp((phys.ARM_C2 * (T - phys.T0)) / (T - phys.T0 + phys.ARM_C3))

