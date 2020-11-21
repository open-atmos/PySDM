"""
Created at 17.02.2020
"""

import numpy as np

from . import conf
import PySDM.physics.constants as const
from numba import float64

from PySDM.physics import _flag
if _flag.DIMENSIONAL_ANALYSIS:
    import PySDM.physics._fake_numba as numba
else:
    import numba


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
    """
    August-Roche-Magnus formula
    """
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


@numba.njit([
    float64(float64),
    float64[:, :](float64[:, :])
], **{**conf.JIT_FLAGS, **{'parallel': False}})
def A(T):
    """
    Koehler curve (expressed in partial pressure)
    """
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


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
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
def dT_i_dt_FF(r, T, p, T_i, dr_dt):
    return 3 / r / const.c_pw * (
        K(r, T, p) / const.rho_w / r * (T - T_i) +  # TODO: K(T) vs. K(Td) ???
        lv(T_i) * dr_dt
    )


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def radius(volume):
    return (volume * 3 / 4 / np.pi) ** (1 / 3)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def dthd_dt(rhod, thd, T, dqv_dt):
    return - lv(T) * dqv_dt / const.c_pd / T * thd * rhod


@numba.njit(float64(float64), **{**conf.JIT_FLAGS, **{'parallel': False}})
def lv(T):
    """
    latent heat of evaporation
    """
    return const.l_tri + (const.c_pv - const.c_pw) * (T - const.T_tri)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def within_tolerance(error_estimate, value, rtol):
    return error_estimate < rtol * np.abs(value)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}} )
def bisec(minfun, a, interval, args, rtol):
    b = a + interval

    fa = minfun(a, *args)
    fb = minfun(b, *args)

    counter = 0
    while fa * fb > 0:
        counter += 1
        if counter > 100:
            raise RuntimeError("Cannot find interval!")
        b = a + interval * 2**counter
        fb = minfun(b, *args)

    if b < a:
        a, b = b, a
        fa, fb = fb, fa

    fb = None

    while True:
        x_new = (a + b) / 2
        if within_tolerance(error_estimate=(b - a), value=x_new, rtol=rtol):
            break

        f_new = minfun(x_new, *args)
        if f_new * fa > 0:
            a = x_new
            fa = f_new
        else:
            b = x_new
    return x_new
