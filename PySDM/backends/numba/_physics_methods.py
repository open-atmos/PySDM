import numpy as np
from numba import float64
import PySDM.simulation.physics.constants as const
from PySDM.backends.numba import conf

from PySDM.simulation.physics import _flag
if _flag.DIMENSIONAL_ANALYSIS:
    import PySDM.simulation.physics._fake_numba as numba
else:
    import numba


class PhysicsMethods:
    # TODO: move somewhere
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def explicit_in_space(omega, c_l, c_r):
        return c_l * (1 - omega) + c_r * omega

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def implicit_in_space(omega, c_l, c_r):
        # see eqs 14-16 in Arabas et al. 2015 (libcloudph++)
        dC = c_r - c_l
        return (omega * dC + c_l) / (1 - dC)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def temperature_pressure_RH(rhod, thd, qv):
        return temperature_pressure_RH(rhod, thd, qv)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def radius(volume):
        return radius(volume)

    @staticmethod
    @numba.njit(
        [float64(float64, float64, float64, float64, float64, float64)],
        **{**conf.JIT_FLAGS, **{'parallel': False}}
    )
    def dlnv_dt(lnv, dr_dt):
        return dlnv_dt(lnv, dr_dt)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def dr_dt_MM(r, T, p, RH, kp, rd):
        return dr_dt_MM(r, T, p, RH, kp, rd)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def dr_dt_FF(r, T, p, qv, kp, rd, T_i):
        return dr_dt_FF(r, T, p, qv, kp, rd, T_i)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def dthd_dt(rhod, thd, T, dqv_dt):
        return dthd_dt(rhod, thd, T, dqv_dt)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def temperature_pressure_RH(rhod, thd, qv):
    # equivalent to eqs A11 & A12 in libcloudph++ 1.0 paper
    exponent = const.Rd / const.c_pd
    pd = np.power((rhod * const.Rd * thd) / const.p1000 ** exponent, 1 / (1 - exponent))
    T = thd * (pd / const.p1000) ** exponent

    R = const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)
    p = rhod * (1 + qv) * R * T

    RH = (p - pd) / pvs(T)

    return T, p, RH


@numba.njit(float64(float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def pvs(T):
    # August-Roche-Magnus formula
    return const.ARM_C1 * np.exp((const.ARM_C2 * (T - const.T0)) / (T - const.T0 + const.ARM_C3))


@numba.njit(float64(float64, float64, float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def _mix(q, dry, wet):
    return wet / (1 / q + 1) + dry / (1 + q)


@numba.njit(float64(float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def c_p(q):
    return _mix(q, const.c_pd, const.c_pv)


@numba.njit(float64(float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def R(q):
    return _mix(q, const.Rd, const.Rv)


''' latent heat of evaporation '''
@numba.njit(float64(float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def lv(T):
    return const.l_tri + (const.c_pv - const.c_pw) * (T - const.T_tri)


@numba.njit(float64(float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def lambdaD(T):
    return const.D0 / np.sqrt(2 * const.Rv * T)


@numba.njit(float64(float64, float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def lambdaK(T, p):
    return (4 / 5) * const.K0 * T / p / np.sqrt(2 * const.Rd * T)


@numba.njit(float64(float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def beta(Kn):
    return (1 + Kn) / (1 + 1.71 * Kn + 1.33 * Kn * Kn)


@numba.njit(float64(float64, float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def D(r, T):
    Kn = lambdaD(T) / r  # TODO: optional
    return const.D0 * beta(Kn)


@numba.njit(float64(float64, float64, float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def K(r, T, p):
    Kn = lambdaK(T, p) / r
    return const.K0 * beta(Kn)


@numba.njit(float64(float64, float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def Fd(T, D):
    return const.rho_w * const.Rv * T / D / pvs(T)


@numba.njit(float64(float64, float64, float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def Fk(T, K, lv):
    return const.rho_w * lv / K / T * (lv / const.Rv / T - 1)


''' Koehler curve (expressed in partial pressure) '''
@numba.njit([
    float64(float64),
    float64[:, :](float64[:, :])
], **{**conf.JIT_FLAGS, **{'parallel': False}})
def A(T):
    return 2 * const.sgm / const.Rv / T / const.rho_w


@numba.njit(float64(float64, float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def B(kp, rd):
    return kp * rd ** 3


@numba.njit([
    float64(float64, float64, float64),
    float64[:, :](float64, float64[:], float64[:, :])
], **{**conf.JIT_FLAGS, **{'parallel': False}})
def r_cr(kp, rd, T):
    # critical radius
    return np.sqrt(3 * kp * rd ** 3 / A(T))


@numba.jit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def RH_eq(r, T, kp, rd):
    return 1 + A(T) / r - B(kp, rd) / r ** 3


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def dr_dt_MM(r, T, p, RH, kp, rd):
    nom = (RH - RH_eq(r, T, kp, rd))
    den = (
            Fd(T, D(r, T)) +
            Fk(T, K(r, T, p), lv(T))
    )
    return 1 / r * nom / den


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def dr_dt_FF(r, T, p, qv, kp, rd, T_i):
    rho_v = p * qv / R(qv) / T
    rho_eq = pvs(T_i) * RH_eq(r, T_i, kp, rd) / const.Rv / T_i
    return D(r, T_i) / const.rho_w / r * (rho_v - rho_eq)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def radius(volume):
    return (volume * 3 / 4 / np.pi) ** (1 / 3)


@numba.njit(
    [float64(float64, float64, float64, float64, float64, float64)],
    **{**conf.JIT_FLAGS, **{'parallel': False}}
)
def dlnv_dt(lnv, dr_dt):
    r = radius(np.exp(lnv))  # TODO: abstract out lnv
    return 3 / r * dr_dt


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def dthd_dt(rhod, thd, T, dqv_dt):
    return - lv(T) * dqv_dt / const.c_pd / T * thd * rhod

