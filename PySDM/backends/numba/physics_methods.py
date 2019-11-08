import numpy as np
import numba
import PySDM.simulation.phys as const

p1000 = const.mgn(const.p1000)
Rd = const.mgn(const.Rd)
Rv = const.mgn(const.Rv)
c_pd = const.mgn(const.c_pd)
ARM_C1 = const.mgn(const.ARM_C1)
ARM_C2 = const.mgn(const.ARM_C2)
ARM_C3 = const.mgn(const.ARM_C3)
T0 = const.mgn(const.T0)


class PhysicsMethods:

    @staticmethod
    @numba.njit()
    def temperature_pressure_RH(rhod, thd, qv):
        # equivalent to eqs A11 & A12 in libcloudph++ 1.0 paper
        kappa = Rd / c_pd
        pd = np.power((rhod * Rd * thd) / p1000 ** kappa, 1 / (1 - kappa))
        T = thd * (pd / p1000) ** kappa

        R = Rv / (1 / qv + 1) + Rd / (1 + qv)
        p = rhod * (1 + qv) * R * T

        # August-Roche-Magnus formula
        pvs = ARM_C1 * np.exp((ARM_C2 * (T - T0)) / (T - T0 + ARM_C3))
        RH = (p - pd) / pvs

        return T, p, RH

    # TODO: move somewhere
    @staticmethod
    @numba.njit()
    def explicit_in_space(omega, c_l, c_r):
        return c_l * (1 - omega) + c_r * omega

    @staticmethod
    @numba.njit()
    def implicit_in_space(omega, c_l, c_r):
        # see eqs 14-16 in Arabas et al. 2015 (libcloudph++)
        dC = c_r - c_l
        return (omega * dC + c_l) / (1 - dC)