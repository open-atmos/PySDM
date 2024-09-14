"""
Physical constants and formulae (mostly one-liners)
  that can be automatically either njit-ted or translated to C (in contrast to
  more complex code that resides in backends). Note that any code here should
  only use constants defined in `PySDM.physics.constants_defaults` and accessible
  through the first `const` argument of each formula, for the following reasons:
  - it makes it possible for a user to override value of any constant by
    passing a `constants` dictionary to the `__init__` of `Formulae`;
  - it enforces floating-point cast on all constant values making the code behave
    in the same way on both CPU and GPU backends (yes, please use `const.ONE/const.TWO`
    instead of `1/2`);
  - it enables dimensional analysis logic if the code in question is embedded
    in `with DimensionalAnalysis:` block - allows checking physical unit consistency
    within unit tests (disable by default, no runtime overhead);
  - last but not least, it requires all the constants to be named
    (thus resulting in more readable, and more reusable code).
"""

from . import (
    condensation_coordinate,
    constants_defaults,
    diffusion_kinetics,
    diffusion_thermics,
    drop_growth,
    fragmentation_function,
    freezing_temperature_spectrum,
    heterogeneous_ice_nucleation_rate,
    hydrostatics,
    hygroscopicity,
    impl,
    isotope_equilibrium_fractionation_factors,
    isotope_meteoric_water_line_excess,
    isotope_ratio_evolution,
    isotope_diffusivity_ratios,
    isotope_relaxation_timescale,
    latent_heat,
    optical_albedo,
    optical_depth,
    particle_advection,
    particle_shape_and_density,
    saturation_vapour_pressure,
    state_variable_triplet,
    surface_tension,
    trivia,
    ventilation,
    air_dynamic_viscosity,
    terminal_velocity,
    bulk_phase_partitioning,
)
from .constants import convert_to, in_unit, si
