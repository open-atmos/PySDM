"""
Created at 13.05.2020
"""

import numpy as np
from .r_wet_init import r_wet_init_impl
from .multiplicities import n_init
from .temperature_init import temperature_init
from PySDM.physics import formulae as phys


def moist_environment_init(
        attributes,
        environment,
        spatial_discretisation,
        spectral_discretisation,
        spectrum_per_mass_of_dry_air,
        r_range,
        kappa,
        enable_temperatures=False
):
    with np.errstate(all='raise'):
        positions = spatial_discretisation(environment.mesh.grid, environment.particles.n_sd)
        attributes['cell id'], attributes['cell origin'], attributes['position in cell'] = \
            environment.mesh.cellular_attributes(positions)
        r_dry, n_per_kg = spectral_discretisation(environment.particles.n_sd, spectrum_per_mass_of_dry_air, r_range)
        T = environment['T'].to_ndarray()
        p = environment['p'].to_ndarray()
        RH = environment['RH'].to_ndarray()
        r_wet = r_wet_init_impl(r_dry, T, p, RH, attributes['cell id'], kappa)
        rhod = environment['rhod'].to_ndarray()
        n_per_m3 = n_init(n_per_kg, rhod, environment.mesh, attributes['cell id'])

    if enable_temperatures:
        attributes['temperature'] = temperature_init(environment, attributes['cell id'])
    attributes['n'] = n_per_m3
    attributes['volume'] = phys.volume(radius=r_wet)
    attributes['dry volume'] = phys.volume(radius=r_dry)
