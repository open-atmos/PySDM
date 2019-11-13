"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state import State
from PySDM.simulation import phys
from PySDM.simulation.phys import si
from PySDM.simulation.ambient_air.moist_air import MoistAir
from PySDM.utils import Physics
import numpy as np


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
            # root-finding initiall guess
            a = r_d
            b = phys.mgn(phys.r_cr(kappa, r_d * si.metres, ambient_air.T[cid] * si.kelvins))
            # minimisation
            # TODO
            #   f = minfun(ph, ix, T0, S0, p0, kp, sys.rd[i])
            #   r0.magnitude[i] = root.brentq(f, a, b)

        return r_wet

    def __call__(self, state: State):
        self.ambient_air.sync()

        x = state.get_backend_storage("x")
        rd = state.get_backend_storage("rd")

        for i in state.idx[:state.SD_num]:
            cid = state.cell_id[i]
            r = Physics.x2r(x[i]) * si.metres
            T = self.ambient_air.T[cid] * si.kelvin
            p = self.ambient_air.p[cid] * si.pascal
            S = self.ambient_air.RH - 1
            kp = self.kappa
            rd = rd[i] * si.metres

            r_new = r + self.dt * self.dr_dt_MM(r, T, p, S, kp, rd)

            x[i] = Physics.r2x(phys.mgn(r_new))**3

        # TODO: update drop radii
        #       update fields due to condensation/evaporation
        #       ensure the above does include droplets that precipitated out of the domain

    @staticmethod
    def dr_dt_MM(r, T, p, S, kp, rd):
        a = (S - Condensation.A(T) / r + Condensation.B(kp, rd) / r ** 3)
        b = Condensation.Fd(T, Condensation.D(r, T)) + Condensation.Fk(T, Condensation.K(r, T, p), Condensation.lv(T))
        result = 1 / r * a / b

    # def mgn(self, q, u=None):
    #     if u is None: return q.to_base_units().magnitude
    #     return (q / u).to(self.si.dimensionless).magnitude
    #
    # def dot(self, a, b):
    #     return np.dot(a.magnitude, b.magnitude) * a.units * b.units
    #
    # def dp_dt(self, p, T, w, q):
    #     # pressure deriv. (hydrostatic)
    #     return -(p / self.R(q) / T) * phys.g * w
    #
    # def dT_dt(self, T, p, dp_dt, q, dq_dt):
    #     # temperature deriv. (adiabatic)
    #     return (T * self.R(q) / p * dp_dt - self.lv(T) * dq_dt) / self.c_p(q)
    #
    # def dS_dt(self, T, p, dp_dt, q, dq_dt, S):
    #     lv = self.lv(T)
    #     cp = self.c_p(q)
    #     return (S + 1) * (
    #             dp_dt / p * (1 - lv * self.R(q) / cp / self.Rv / T) +
    #             dq_dt * (lv ** 2 / cp / self.Rv / T ** 2 + 1 / (q + q ** 2 / self.eps))
    #     )
    #
    # def q(self, q0, n, r):
    #     return q0 - 4 / 3 * self.pi * self.dot(n, r ** 3) * self.rho_w
    #
    # def dq_dt(self, n, r, dr_dt):
    #     # specific humidity deriv.
    #     return -4 * self.pi * np.sum(n * r ** 2 * dr_dt) * self.rho_w
    #
    # def dr_dt_Fick(self, r, T, S, kp, rd, Td):
    #     rho_v = (S + 1) * self.pvs(T) / self.Rv / T
    #     rho_eq = self.pvs(Td) * (1 + self.A(T) / r - self.B(kp, rd) / r ** 3) / self.Rv / Td
    #     D = self.D(r, Td)  # TODO: K(T) vs. K(Td) ???
    #     return D / r / self.rho_w * (rho_v - rho_eq)
    #
    # def dTd_dt(self, r, T, p, Td, dr_dt):
    #     return 3 / r / self.c_pw * (
    #             self.K(r, T, p) / self.rho_w / r * (T - Td) +  # TODO: K(T) vs. K(Td) ???
    #             self.lv(Td) * dr_dt
    #     )
    #
    # def RH(self, T, p, q):
    #     # RH from mixing ratio, temperature and pressure
    #     return p / (1 + self.eps / q) / self.pvs(T)
    #
    # def mix(self, q, dry, wet):
    #     return wet / (1 / q + 1) + dry / (1 + q)
    #
    # def c_p(self, q):
    #     return self.mix(q, self.c_pd, self.c_pv)
    #
    # def R(self, q):
    #     return self.mix(q, self.Rd, self.Rv)
    #
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

    # def bpt(self, p):
    #     # inverse of the above
    #     bpt_log = np.log(p / self.ARM_C1) / self.ARM_C2
    #     return self.ARM_C3 * bpt_log / (1 - bpt_log) + self.T0
    #
    # def T(self, S, p, q):
    #     return self.bpt(p / (S + 1) / (1 + self.eps / q))

