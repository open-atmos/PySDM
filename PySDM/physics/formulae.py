"""
Crated at 2019
"""

from PySDM.backends.numba import conf
from functools import lru_cache
from PySDM.physics.impl import flag

if flag.DIMENSIONAL_ANALYSIS:
    # TODO #492 - with all formulae refactored, this will not be needed anymore!
    from PySDM.physics.impl.fake_numba import njit
    _formula = njit
else:
    import numba
    def _formula(func=None, **kw):
        if func is None:
            return numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'inline': 'always', **kw}})
        else:
            return numba.njit(func, **{**conf.JIT_FLAGS, **{'parallel': False, 'inline': 'always', **kw}})

from PySDM.physics import constants as const
import numpy as np
from PySDM.physics.constants import R_str
from pystrict import strict
from PySDM import physics
from PySDM.physics.trivia import Trivia


def _boost(obj, fastmath):
    if not flag.DIMENSIONAL_ANALYSIS:
        for item in dir(obj):
            if item.startswith('__'):
                continue
            attr = getattr(obj, item)
            if callable(attr):
                setattr(obj, item, _formula(attr, fastmath=fastmath))
    return obj


def _pick(value: str, choices: dict):
    for name, cls in choices.items():
        if name == value:
            return cls()
    raise ValueError(f"Unknown setting: '{value}';, choices are: {tuple(choices.keys())}")


def _choices(module):
    return dict([(name, cls) for name, cls in module.__dict__.items() if isinstance(cls, type)])


@lru_cache()
def _magick(value, module, fastmath):
    return _boost(_pick(value, _choices(module)), fastmath)


@strict
class Formulae:
    def __init__(self, *,
                 fastmath: bool = True,
                 condensation_coordinate: str = 'VolumeLogarithm',
                 saturation_vapour_pressure: str = 'FlatauWalkoCotton',
                 latent_heat: str = 'Kirchhoff',
                 hygroscopicity: str = 'KappaKoehler',
                 drop_growth: str = 'MaxwellMason',
                 surface_tension: str = 'Constant',
                 diffusion_kinetics: str = 'FuchsSutugin',
                 ventilation: str = 'Neglect',
                 state_variable_triplet: str = 'RhodThdQv'
                 ):
        self.fastmath = fastmath
        self.trivia = _boost(Trivia(), fastmath)
        self.condensation_coordinate = _magick(condensation_coordinate, physics.condensation_coordinate, fastmath)
        self.saturation_vapour_pressure = _magick(saturation_vapour_pressure, physics.saturation_vapour_pressure, fastmath)
        self.latent_heat = _magick(latent_heat, physics.latent_heat, fastmath)
        self.hygroscopicity = _magick(hygroscopicity, physics.hygroscopicity, fastmath)
        self.drop_growth = _magick(drop_growth, physics.drop_growth, fastmath)
        self.surface_tension = _magick(surface_tension, physics.surface_tension, fastmath)
        self.diffusion_kinetics = _magick(diffusion_kinetics, physics.diffusion_kinetics, fastmath)
        self.ventilation = _magick(ventilation, physics.ventilation, fastmath)
        self.state_variable_triplet = _magick(state_variable_triplet, physics.state_variable_triplet, fastmath)

    def __str__(self):
        description = []
        for attr in dir(self):
            if not attr.startswith('_'):
                description.append(f"{attr}: {getattr(self, attr).__class__.__name__}")
        return ', '.join(description)


@_formula
def R(q):
    return _mix(q, const.Rd, const.Rv)


@_formula
def c_p(q):
    return _mix(q, const.c_pd, const.c_pv)


@_formula
def dthd_dt(rhod, thd, T, dqv_dt, lv):
    return - lv * dqv_dt / const.c_pd / T * thd * rhod


@_formula
def temperature_pressure_pv(rhod, thd, qv):
    # equivalent to eqs A11 & A12 in libcloudph++ 1.0 paper
    exponent = const.Rd / const.c_pd
    pd = np.power((rhod * const.Rd * thd) / const.p1000 ** exponent, 1 / (1 - exponent))
    T = thd * (pd / const.p1000) ** exponent
    R = const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)
    p = rhod * (1 + qv) * R * T
    return T, p, p - pd


@_formula
def radius(volume):
    return (volume * 3 / 4 / np.pi) ** (1 / 3)


@_formula
def volume(radius):
    return 4 / 3 * np.pi * radius ** 3


@_formula
def pH2H(pH):
    return 10**(-pH) * 1e3


@_formula
def th_dry(th_std, qv):
    return th_std * np.power(1 + qv / const.eps, const.Rd / const.c_pd)


@_formula
def th_std(p, T):
    return T * (const.p1000 / p)**(const.Rd / const.c_pd)


class MoistAir:
    @staticmethod
    def rhod_of_rho_qv(rho, qv):
        return rho / (1 + qv)

    @staticmethod
    def rho_of_rhod_qv(rhod, qv):
        return rhod * (1 + qv)

    @staticmethod
    def p_d(p, qv):
        return p * (1 - 1 / (1 + const.eps / qv))

    @staticmethod
    def rhod_of_pd_T(pd, T):
        return pd / const.Rd / T

    @staticmethod
    def rho_of_p_qv_T(p, qv, T):
        return p / R(qv) / T


class ThStd:
    @staticmethod
    def rho_d(p, qv, theta_std):
        kappa = const.Rd / const.c_pd
        pd = MoistAir.p_d(p, qv)
        rho_d = pd / (np.power(p / const.p1000, kappa) * const.Rd * theta_std)
        return rho_d


class Hydrostatic:
    @staticmethod
    def drho_dz(g, p, T, qv, lv, dql_dz=0):
        rho = MoistAir.rho_of_p_qv_T(p, qv, T)
        Rq = R(qv)
        cp = c_p(qv)
        return (g / T * rho * (Rq / cp - 1) - p * lv / cp / T**2 * dql_dz) / Rq

    @staticmethod
    def p_of_z_assuming_const_th_and_qv(g, p0, thstd, qv, z):
        kappa = const.Rd / const.c_pd
        z0 = 0
        arg = np.power(p0/const.p1000, kappa) - (z-z0) * kappa * g / thstd / R(qv)
        return const.p1000 * np.power(arg, 1/kappa)


def explicit_euler(y, dt, dy_dt):
    y += dt * dy_dt


@_formula
def mole_fraction_2_mixing_ratio(mole_fraction, specific_gravity):
    return specific_gravity * mole_fraction / (1 - mole_fraction)


@_formula
def mixing_ratio_2_mole_fraction(mixing_ratio, specific_gravity):
    return mixing_ratio / (specific_gravity + mixing_ratio)


@_formula
def mixing_ratio_2_partial_pressure(mixing_ratio, specific_gravity, pressure):
    return pressure * mixing_ratio / (mixing_ratio + specific_gravity)


@_formula
def _mix(q, dry, wet):
    return wet / (1 / q + 1) + dry / (1 + q)


@_formula
def lambdaD(T):
    return const.D0 / np.sqrt(2 * const.Rv * T)


@_formula
def lambdaK(T, p):
    return (4 / 5) * const.K0 * T / p / np.sqrt(2 * const.Rd * T)


@_formula
def beta(Kn):
    return (1 + Kn) / (1 + 1.71 * Kn + 1.33 * Kn * Kn)


@_formula
def D(r, T):
    Kn = lambdaD(T) / r  # TODO #57 optional
    return const.D0 * beta(Kn)


@_formula
def K(r, T, p):
    Kn = lambdaK(T, p) / r
    return const.K0 * beta(Kn)


@_formula
def within_tolerance(error_estimate, value, rtol):
    return error_estimate < rtol * np.abs(value)


@_formula
def H2pH(H):
    return -np.log10(H * 1e-3)


@_formula
def vant_hoff(K, dH, T, *, T_0):
    return K * np.exp(-dH / R_str * (1 / T - 1/T_0))


@_formula
def tdep2enthalpy(tdep):
    return -tdep * R_str


@_formula
def arrhenius(A, Ea, T):
    return A * np.exp(-Ea / (R_str * T))
