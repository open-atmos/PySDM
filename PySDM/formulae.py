"""
Logic for enabling common CPU/GPU physics formulae code
"""

import inspect
import math
import numbers
import re
import warnings
from collections import namedtuple
from functools import lru_cache, partial, cached_property
from types import SimpleNamespace
from typing import Optional

import numba
import numpy as np
import pint
from numba.core.errors import NumbaExperimentalFeatureWarning

from PySDM import physics
from PySDM.backends.impl_numba import conf
from PySDM.dynamics.terminal_velocity import GunnKinzer1949, PowerSeries, RogersYau
from PySDM.dynamics.terminal_velocity.gunn_and_kinzer import TpDependent


class Formulae:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-locals
        self,
        *,
        constants: Optional[dict] = None,
        seed: int = None,
        fastmath: bool = True,
        condensation_coordinate: str = "VolumeLogarithm",
        saturation_vapour_pressure: str = "FlatauWalkoCotton",
        latent_heat: str = "Kirchhoff",
        hygroscopicity: str = "KappaKoehlerLeadingTerms",
        drop_growth: str = "Mason1971",
        surface_tension: str = "Constant",
        diffusion_kinetics: str = "FuchsSutugin",
        diffusion_thermics: str = "Neglect",
        ventilation: str = "Neglect",
        state_variable_triplet: str = "LibcloudphPlusPlus",
        particle_advection: str = "ImplicitInSpace",
        hydrostatics: str = "Default",
        freezing_temperature_spectrum: str = "Null",
        heterogeneous_ice_nucleation_rate: str = "Null",
        fragmentation_function: str = "AlwaysN",
        isotope_equilibrium_fractionation_factors: str = "Null",
        isotope_meteoric_water_line_excess: str = "Null",
        isotope_ratio_evolution: str = "Null",
        isotope_diffusivity_ratios: str = "Null",
        isotope_relaxation_timescale: str = "Null",
        optical_albedo: str = "Null",
        optical_depth: str = "Null",
        particle_shape_and_density: str = "LiquidSpheres",
        terminal_velocity: str = "GunnKinzer1949",
        air_dynamic_viscosity: str = "ZografosEtAl1987",
        bulk_phase_partitioning: str = "Null",
        handle_all_breakups: bool = False,
    ):
        # initialisation of the fields below is just to silence pylint and to enable code hints
        # in PyCharm and alike, all these fields are later overwritten within this ctor
        self.optical_albedo = optical_albedo
        self.optical_depth = optical_depth
        self.condensation_coordinate = condensation_coordinate
        self.saturation_vapour_pressure = saturation_vapour_pressure
        self.hygroscopicity = hygroscopicity
        self.drop_growth = drop_growth
        self.surface_tension = surface_tension
        self.diffusion_kinetics = diffusion_kinetics
        self.latent_heat = latent_heat
        self.diffusion_thermics = diffusion_thermics
        self.ventilation = ventilation
        self.state_variable_triplet = state_variable_triplet
        self.particle_advection = particle_advection
        self.hydrostatics = hydrostatics
        self.freezing_temperature_spectrum = freezing_temperature_spectrum
        self.heterogeneous_ice_nucleation_rate = heterogeneous_ice_nucleation_rate
        self.fragmentation_function = fragmentation_function
        self.isotope_equilibrium_fractionation_factors = (
            isotope_equilibrium_fractionation_factors
        )
        self.isotope_meteoric_water_line_excess = isotope_meteoric_water_line_excess
        self.isotope_ratio_evolution = isotope_ratio_evolution
        self.isotope_diffusivity_ratios = isotope_diffusivity_ratios
        self.isotope_relaxation_timescale = isotope_relaxation_timescale
        self.particle_shape_and_density = particle_shape_and_density
        self.air_dynamic_viscosity = air_dynamic_viscosity
        self.terminal_velocity = terminal_velocity
        self.bulk_phase_partitioning = bulk_phase_partitioning

        self._components = tuple(
            i
            for i in dir(self)
            if not i.startswith("__") and i not in ("flatten", "get_constant")
        )

        constants_defaults = {
            k: getattr(defaults, k)
            for defaults in (physics.constants, physics.constants_defaults)
            for k in dir(defaults)
            if isinstance(
                getattr(defaults, k), (numbers.Number, pint.Quantity, pint.Unit)
            )
        }

        physics.constants_defaults.compute_derived_values(constants_defaults)
        if constants is not None:
            for key in constants:
                if key not in constants_defaults:
                    raise ValueError(
                        f"constant override provided for unknown key: {key}"
                    )

        constants_defaults = {**constants_defaults, **(constants or {})}
        physics.constants_defaults.compute_derived_values(constants_defaults)

        constants = namedtuple("Constants", tuple(constants_defaults.keys()))(
            **constants_defaults
        )
        self.constants = constants
        self.seed = seed or physics.constants.default_random_seed
        self.fastmath = fastmath
        self.handle_all_breakups = handle_all_breakups
        dimensional_analysis = physics.impl.flag.DIMENSIONAL_ANALYSIS

        self.trivia = _magick(
            "Trivia", physics.trivia, fastmath, constants, dimensional_analysis
        )

        # each `component` corresponds to one subdirectory of PySDM/physics
        for component in self._components:
            setattr(
                self,
                component,
                _magick(
                    value=getattr(self, component),
                    module=getattr(physics, component),
                    fastmath=fastmath,
                    constants=constants,
                    dimensional_analysis=dimensional_analysis,
                ),
            )

        # TODO #348
        self.terminal_velocity_class = {
            "GunnKinzer1949": GunnKinzer1949,
            "RogersYau": RogersYau,
            "TpDependent": TpDependent,
            "PowerSeries": PowerSeries,
        }[terminal_velocity]

    def __str__(self):
        description = []
        for attr in dir(self):
            if not attr.startswith("_") and attr != "flatten":
                attr_value = getattr(self, attr)
                if attr_value.__class__ in (bool, int, float):
                    value = attr_value
                elif attr_value.__class__.__name__ == "Constants":
                    value = str(attr_value)
                elif attr_value.__class__ in (SimpleNamespace,):
                    value = attr_value.__name__
                else:
                    value = attr_value.__class__.__name__
                description.append(f"{attr}: {value}")
        return ", ".join(description)

    @cached_property
    def flatten(self):
        """returns a "flattened" representation providing access to all formulae from within
        one Numba-JIT-usable named tuple, e.g. with obj.latent_heat__lv(T)"""
        functions = {}
        for component in ["trivia"] + list(self._components):
            for item in dir(getattr(self, component)):
                attr = getattr(getattr(self, component), item)
                if not item.startswith("__") and callable(attr):
                    functions[component + "__" + item] = attr
        for attr in ("constants", "fastmath"):
            functions[attr] = getattr(self, attr)
        return namedtuple("FlattenedFormulae", functions.keys())(**functions)

    def get_constant(self, key: str):
        """getter-like method for cases where using the `constants` named tuple is not possible
        (e.g., if calling from a language which does not support named tuples)"""
        return getattr(self.constants, key)


def _formula(func, constants, dimensional_analysis, **kw):
    parameters_keys = tuple(inspect.signature(func).parameters.keys())
    special_params = ("_", "const")

    if dimensional_analysis:
        first_param = parameters_keys[0]
        if first_param in special_params:
            return partial(func, constants)
        return func

    source = "class _:\n" + "".join(inspect.getsourcelines(func)[0])
    source = re.sub(r"\(\n\s+", "(", source)
    source = re.sub(r"\n\s+\):", "):", source)
    loc = {}
    for arg_name in special_params:
        source = source.replace(
            f"def {func.__name__}({arg_name},", f"def {func.__name__}("
        )

    extras = func.__extras if hasattr(func, "__extras") else {}
    exec(  # pylint:disable=exec-used
        source, {"const": constants, "np": np, "math": math, **extras}, loc
    )

    n_params = len(parameters_keys) - (1 if parameters_keys[0] in special_params else 0)
    function = getattr(loc["_"], func.__name__)
    if hasattr(func, "__vectorize"):
        vectorizer = (
            np.vectorize
            if numba.config.DISABLE_JIT  # pylint: disable=no-member
            else numba.vectorize(
                "float64(" + ",".join(["float64"] * n_params) + ")",
                target="cpu",
                nopython=True,
                **{
                    k: v
                    for k, v in conf.JIT_FLAGS.items()
                    if k not in ("parallel", "error_model")
                },
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
            return vectorizer(function)
    return numba.njit(
        getattr(loc["_"], func.__name__),
        **{
            **conf.JIT_FLAGS,
            **{"parallel": False, "inline": "always", "cache": False, **kw},
        },
    )


def _boost(obj, fastmath, constants, dimensional_analysis):
    """returns JIT-compiled, `c_inline`-equipped formulae with the constants catalogue attached"""
    formulae = {"__name__": obj.__name__}
    for item in dir(obj):
        attr = getattr(obj, item)
        if item.startswith("__") or not callable(attr):
            pass
        else:
            formula = _formula(
                attr,
                constants=constants,
                fastmath=fastmath,
                dimensional_analysis=dimensional_analysis,
            )
            setattr(
                formula, "c_inline", partial(_c_inline, constants=constants, fun=attr)
            )
            formulae[attr.__name__] = formula
    return SimpleNamespace(**formulae)


def _c_inline(fun, return_type=None, constants=None, **args):
    real_t = "real_type"
    return_type = return_type or real_t
    prae = r"([,+\-*/( ]|^)"
    post = r"([ )/*\-+,]|$)"
    real_fmt = ".32g"
    source = ""
    for line in inspect.getsourcelines(fun)[0]:
        stripped = line.strip()
        if stripped.startswith("@"):
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith("def "):
            continue
        if stripped.endswith(","):
            stripped += " "
        source += stripped
    source = source.replace("np.power(", "np.pow(")
    source = source.replace("np.arctanh(", "atanh(")
    source = source.replace("np.arcsinh(", "asinh(")
    source = source.replace("np.minimum(", "min(")
    source = source.replace("np.maximum(", "max(")
    for pkg in ("np", "math"):
        source = source.replace(f"{pkg}.", "")
    source = source.replace(", )", ")")
    source = re.sub(pattern="^return ", repl="", string=source)
    for arg in inspect.signature(fun).parameters:
        if arg not in ("_", "const"):
            source = re.sub(
                pattern=f"{prae}({arg}){post}",
                repl=f"\\1({real_t})({args[arg]})\\3",
                string=source,
            )
    source = re.sub(
        pattern=f"{prae}const\\.([^\\d\\W]\\w*]*){post}",
        repl="\\1(" + real_t + ")({constants.\\2:" + real_fmt + "})\\3",
        string=source,
    )
    assert constants
    source = eval(f'f"""{source}"""')  # pylint: disable=eval-used
    return f"({return_type})({source})"


def _pick(value: str, choices: dict, constants: namedtuple):
    """
    selects a given physics logic and instantiates it passing the constants catalogue;
    `value` is expected to be string containing a plus-separated list of class names
    """

    obj = None
    if "+" not in value:
        for name, cls in choices.items():
            if name == value:
                obj = cls(constants)
                break
            name = value
    else:
        parent_classes = []
        for name in value.split("+"):
            if name not in choices:
                parent_classes.clear()
                break
            parent_classes.append(choices[name])

        if len(parent_classes) > 0:

            class Cls(*parent_classes):  # pylint: disable=too-few-public-methods
                def __init__(self, const):
                    for cls in parent_classes:
                        cls.__init__(self, const)

            obj = Cls(constants)

    if obj is None:
        raise ValueError(
            f"Unknown setting: {name}; choices are: {', '.join(choices.keys())}"
        )

    obj.__name__ = value  # pylint: disable=attribute-defined-outside-init
    return obj


def _choices(module):
    return {name: cls for name, cls in module.__dict__.items() if isinstance(cls, type)}


@lru_cache()
def _magick(value, module, fastmath, constants, dimensional_analysis):
    """
    boosts (`PySDM.formulae.Formulae._boost`) the selected physics logic
    """
    return _boost(
        _pick(value, _choices(module), constants),
        fastmath,
        constants,
        dimensional_analysis,
    )
