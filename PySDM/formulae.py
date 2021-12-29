"""
Logic for enabling common CPU/GPU physics formulae code
"""
import inspect
from typing import Optional
import re
from functools import lru_cache, partial
from collections import namedtuple
import numbers
import numba
import numpy as np
from PySDM import physics
from PySDM.backends.impl_numba import conf


def _formula(func=None, constants=None, **kw):
    source = "class _:\n"
    for line in inspect.getsourcelines(func)[0]:
        source += f"{line}\n"
    loc = {}
    for arg_name in ('_', 'const'):
        source = source.replace(f'def {func.__name__}({arg_name},', f'def {func.__name__}(')
    if func.__name__ == 'volume':
        print(" TEMP ", source)
    exec(  # pylint:disable=exec-used
        source,
        {'const': constants, 'np': np},
        loc
    )
    return numba.njit(
        getattr(loc['_'], func.__name__),
        **{
            **conf.JIT_FLAGS,
            **{'parallel': False, 'inline': 'always', 'cache': False, **kw}
        }
    )


def _boost(obj, fastmath, constants):
    if not physics.impl.flag.DIMENSIONAL_ANALYSIS:
        for item in dir(obj):
            if item.startswith('__'):
                continue
            attr = getattr(obj, item)
            if callable(attr):
                formula = _formula(attr, constants=constants, fastmath=fastmath)
                setattr(obj, item, formula)
                setattr(
                    getattr(obj, item),
                    'c_inline',
                    partial(_c_inline, constants=constants, fun=attr)
                )
    return obj


def _c_inline(fun, return_type=None, constants=None, **args):
    real_t = 'real_type'
    return_type = return_type or real_t
    prae = r"([,+\-*/( ]|^)"
    post = r"([ )/*\-+,]|$)"
    real_fmt = ".32g"
    source = ''
    for line in inspect.getsourcelines(fun)[0]:
        stripped = line.strip()
        if stripped.startswith('@'):
            continue
        if stripped.startswith('//'):
            continue
        if stripped.startswith('def '):
            continue
        source += stripped
    # source = source.replace("power(", "pow(")
    source = re.sub("^return ", "", source)
    for arg in inspect.signature(fun).parameters:
        if arg not in ('_', 'const'):
            source = re.sub(f"{prae}({arg}){post}", f"\\1({real_t})({args[arg]})\\3", source)
    source = re.sub(
        f"{prae}const\\.([^\\d\\W]\\w*]*){post}",
        "\\1(" + real_t + ")({constants.\\2:" + real_fmt + "})\\3",
        source
    )
    assert constants
    source = eval(f'f"""{source}"""')  # pylint: disable=eval-used
    return f'({return_type})({source})'


def _pick(value: str, choices: dict, constants: namedtuple):
    for name, cls in choices.items():
        if name == value:
            return cls(constants)
    raise ValueError(f"Unknown setting: '{value}'; choices are: {tuple(choices.keys())}")


def _choices(module):
    return {name: cls for name, cls in module.__dict__.items() if isinstance(cls, type)}


@lru_cache()
def _magick(value, module, fastmath, constants):
    return _boost(_pick(value, _choices(module), constants), fastmath, constants)


class Formulae:
    def __init__(self, *,
                 constants: Optional[dict] = None,
                 seed: int = physics.constants.default_random_seed,
                 fastmath: bool = True,
                 condensation_coordinate: str = 'VolumeLogarithm',
                 saturation_vapour_pressure: str = 'FlatauWalkoCotton',
                 latent_heat: str = 'Kirchhoff',
                 hygroscopicity: str = 'KappaKoehlerLeadingTerms',
                 drop_growth: str = 'MaxwellMason',
                 surface_tension: str = 'Constant',
                 diffusion_kinetics: str = 'FuchsSutugin',
                 diffusion_thermics: str = 'Neglect',
                 ventilation: str = 'Neglect',
                 state_variable_triplet: str = 'RhodThdQv',
                 particle_advection: str = 'ImplicitInSpace',
                 hydrostatics: str = 'Default',
                 freezing_temperature_spectrum: str = 'Null',
                 heterogeneous_ice_nucleation_rate: str = 'Null'
                 ):
        constants_defaults = {
            k: getattr(physics.constants, k)
            for k in dir(physics.constants)
            if isinstance(getattr(physics.constants, k), numbers.Number)
        }
        constants = namedtuple(
            "Constants",
            tuple(constants_defaults.keys())
        )(
            **{**constants_defaults, **(constants or {})}
        )
        self.constants = constants
        self.seed = seed
        self.fastmath = fastmath

        self.trivia = _magick('Trivia', physics.trivia, fastmath, constants)

        self.condensation_coordinate = _magick(
            condensation_coordinate,
            physics.condensation_coordinate, fastmath, constants)
        self.saturation_vapour_pressure = _magick(
            saturation_vapour_pressure,
            physics.saturation_vapour_pressure, fastmath, constants)
        self.latent_heat = _magick(
            latent_heat,
            physics.latent_heat, fastmath, constants)
        self.hygroscopicity = _magick(
            hygroscopicity,
            physics.hygroscopicity, fastmath, constants)
        self.drop_growth = _magick(
            drop_growth,
            physics.drop_growth, fastmath, constants)
        self.surface_tension = _magick(
            surface_tension,
            physics.surface_tension, fastmath, constants)
        self.diffusion_kinetics = _magick(
            diffusion_kinetics,
            physics.diffusion_kinetics, fastmath, constants)
        self.diffusion_thermics = _magick(
            diffusion_thermics,
            physics.diffusion_thermics, fastmath, constants)
        self.ventilation = _magick(
            ventilation,
            physics.ventilation, fastmath, constants)
        self.state_variable_triplet = _magick(
            state_variable_triplet,
            physics.state_variable_triplet, fastmath, constants)
        self.particle_advection = _magick(
            particle_advection,
            physics.particle_advection, fastmath, constants)
        self.hydrostatics = _magick(
            hydrostatics,
            physics.hydrostatics, fastmath, constants)
        self.freezing_temperature_spectrum = _magick(
            freezing_temperature_spectrum,
            physics.freezing_temperature_spectrum, fastmath, constants)
        self.heterogeneous_ice_nucleation_rate = _magick(
            heterogeneous_ice_nucleation_rate,
            physics.heterogeneous_ice_nucleation_rate, fastmath, constants)

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
