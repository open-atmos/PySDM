import numpy as np
from numba import float64
import PySDM.simulation.physics.constants as const

from PySDM.simulation.physics import _flag
if _flag.DIMENSIONAL_ANALYSIS:
    import PySDM.simulation.physics._fake_numba as numba
else:
    import numba


class PhysicsMethods:
    @staticmethod
    @numba.njit()
    def temperature_pressure_RH(rhod, thd, qv):
        # equivalent to eqs A11 & A12 in libcloudph++ 1.0 paper
        exponent = const.Rd / const.c_pd
        pd = np.power((rhod * const.Rd * thd) / const.p1000 ** exponent, 1 / (1 - exponent))
        T = thd * (pd / const.p1000) ** exponent

        R = const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)
        p = rhod * (1 + qv) * R * T

        RH = (p - pd) / pvs(T)

        return T, p, RH

    @staticmethod
    @numba.njit()
    def dr_dt_MM(r, T, p, S, kp, rd):
        nom = (S - A(T) / r + B(kp, rd) / r ** 3)
        den = (
                Fd(T, D(r, T)) +
                Fk(T, K(r, T, p), lv(T))
        )
        result = 1 / r * nom / den
        return result

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


@numba.njit(float64(float64))
def pvs(T):
    # August-Roche-Magnus formula
    return const.ARM_C1 * np.exp((const.ARM_C2 * (T - const.T0)) / (T - const.T0 + const.ARM_C3))


@numba.njit(float64(float64, float64, float64))
def _mix(q, dry, wet):
    return wet / (1 / q + 1) + dry / (1 + q)


@numba.njit(float64(float64))
def c_p(q):
    return _mix(q, const.c_pd, const.c_pv)


@numba.njit(float64(float64))
def R(q):
    return _mix(q, const.Rd, const.Rv)


''' latent heat of evaporation '''
@numba.njit(float64(float64))
def lv(T):
    return const.l_tri + (const.c_pv - const.c_pw) * (T - const.T_tri)


@numba.njit(float64(float64))
def lambdaD(T):
    return const.D0 / np.sqrt(2 * const.Rv * T)


@numba.njit(float64(float64, float64))
def lambdaK(T, p):
    return (4 / 5) * const.K0 * T / p / np.sqrt(2 * const.Rd * T)


@numba.njit(float64(float64))
def beta(Kn):
    return (1 + Kn) / (1 + 1.71 * Kn + 1.33 * Kn * Kn)


@numba.njit(float64(float64, float64))
def D(r, T):
    Kn = lambdaD(T) / r  # TODO: optional
    return const.D0 * beta(Kn)


@numba.njit(float64(float64, float64, float64))
def K(r, T, p):
    Kn = lambdaK(T, p) / r
    return const.K0 * beta(Kn)


@numba.njit(float64(float64, float64))
def Fd(T, D):
    return const.rho_w * const.Rv * T / D / pvs(T)


@numba.njit(float64(float64, float64, float64))
def Fk(T, K, lv):
    return const.rho_w * lv / K / T * (lv / const.Rv / T - 1)


''' Koehler curve (expressed in partial pressure) '''
@numba.njit([
    float64(float64),
    float64[:, :](float64[:, :])
])
def A(T):
    return 2 * const.sgm / const.Rv / T / const.rho_w


@numba.njit(float64(float64, float64))
def B(kp, rd):
    return kp * rd ** 3


@numba.njit([
    float64(float64, float64, float64),
    float64[:, :](float64, float64[:], float64[:, :])
])
def r_cr(kp, rd, T):
    # critical radius
    return np.sqrt(3 * kp * rd ** 3 / A(T))


@numba.njit([
    float64(float64, float64, float64, float64)
])
def mse(T, qv, ql, z):
    return T * (
        const.c_pd +
        const.c_pv * qv +
        const.c_pw * ql
    ) + (1 + qv) * const.g * z