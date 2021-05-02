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
from PySDM import physics


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


class Formulae:
    def __init__(self, *,
                 fastmath: bool = True,
                 condensation_coordinate: str = 'VolumeLogarithm',
                 saturation_vapour_pressure: str = 'FlatauWalkoCotton',
                 latent_heat: str = 'Kirchhoff',
                 hygroscopicity: str = 'KappaKoehlerLeadingTerms',
                 drop_growth: str = 'MaxwellMason',
                 surface_tension: str = 'Constant',
                 diffusion_kinetics: str = 'FuchsSutugin',
                 ventilation: str = 'Neglect',
                 state_variable_triplet: str = 'RhodThdQv',
                 particle_advection: str = 'ImplicitInSpace'
                 ):
        self.fastmath = fastmath
        self.trivia = _magick('Trivia', physics.trivia, fastmath)
        self.condensation_coordinate = _magick(condensation_coordinate, physics.condensation_coordinate, fastmath)
        self.saturation_vapour_pressure = _magick(saturation_vapour_pressure, physics.saturation_vapour_pressure, fastmath)
        self.latent_heat = _magick(latent_heat, physics.latent_heat, fastmath)
        self.hygroscopicity = _magick(hygroscopicity, physics.hygroscopicity, fastmath)
        self.drop_growth = _magick(drop_growth, physics.drop_growth, fastmath)
        self.surface_tension = _magick(surface_tension, physics.surface_tension, fastmath)
        self.diffusion_kinetics = _magick(diffusion_kinetics, physics.diffusion_kinetics, fastmath)
        self.ventilation = _magick(ventilation, physics.ventilation, fastmath)
        self.state_variable_triplet = _magick(state_variable_triplet, physics.state_variable_triplet, fastmath)
        self.particle_advection = _magick(particle_advection, physics.particle_advection, fastmath)

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
def th_std(p, T):
    return T * (const.p1000 / p)**(const.Rd / const.c_pd)


@_formula
def rhod_of_rho_qv(rho, qv):
    return rho / (1 + qv)


@_formula
def rho_of_rhod_qv(rhod, qv):
    return rhod * (1 + qv)


@_formula
def p_d(p, qv):
    return p * (1 - 1 / (1 + const.eps / qv))


@_formula
def rhod_of_pd_T(pd, T):
    return pd / const.Rd / T


@_formula
def rho_of_p_qv_T(p, qv, T):
    return p / R(qv) / T


class ThStd:
    @staticmethod
    @_formula
    def rho_d(p, qv, theta_std):
        kappa = const.Rd / const.c_pd
        pd = p_d(p, qv)
        rho_d = pd / (np.power(p / const.p1000, kappa) * const.Rd * theta_std)
        return rho_d


class Hydrostatic:
    @staticmethod
    @_formula
    def drho_dz(g, p, T, qv, lv, dql_dz=0):
        rho = rho_of_p_qv_T(p, qv, T)
        Rq = R(qv)
        cp = _mix(qv, const.c_pd, const.c_pv)
        return (g / T * rho * (Rq / cp - 1) - p * lv / cp / T**2 * dql_dz) / Rq

    @staticmethod
    @_formula
    def p_of_z_assuming_const_th_and_qv(g, p0, thstd, qv, z):
        kappa = const.Rd / const.c_pd
        z0 = 0
        arg = np.power(p0/const.p1000, kappa) - (z-z0) * kappa * g / thstd / R(qv)
        return const.p1000 * np.power(arg, 1/kappa)


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
def vant_hoff(K, dH, T, *, T_0):
    return K * np.exp(-dH / R_str * (1 / T - 1/T_0))


@_formula
def tdep2enthalpy(tdep):
    return -tdep * R_str


@_formula
def arrhenius(A, Ea, T):
    return A * np.exp(-Ea / (R_str * T))
