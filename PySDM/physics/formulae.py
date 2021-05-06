"""
Crated at 2019
"""

from PySDM.backends.numba import conf
from functools import lru_cache
from PySDM import physics
import numba


def _formula(func=None, **kw):
    if func is None:
        return numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'inline': 'always', **kw}})
    else:
        return numba.njit(func, **{**conf.JIT_FLAGS, **{'parallel': False, 'inline': 'always', **kw}})


def _boost(obj, fastmath):
    if not physics.impl.flag.DIMENSIONAL_ANALYSIS:
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
                 diffusion_thermics: str = 'TracyWelchPorter',
                 ventilation: str = 'Neglect',
                 state_variable_triplet: str = 'RhodThdQv',
                 particle_advection: str = 'ImplicitInSpace',
                 hydrostatics: str = 'Default'
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
        self.diffusion_thermics = _magick(diffusion_thermics, physics.diffusion_thermics, fastmath)
        self.ventilation = _magick(ventilation, physics.ventilation, fastmath)
        self.state_variable_triplet = _magick(state_variable_triplet, physics.state_variable_triplet, fastmath)
        self.particle_advection = _magick(particle_advection, physics.particle_advection, fastmath)
        self.hydrostatics = _magick(hydrostatics, physics.hydrostatics, fastmath)

    def __str__(self):
        description = []
        for attr in dir(self):
            if not attr.startswith('_'):
                if getattr(self, attr).__class__ in (bool, int, float):
                    value = getattr(self, attr)
                else:
                    value = getattr(self, attr).__class__.__name__
                description.append(f"{attr}: {value}")
        return ', '.join(description)
