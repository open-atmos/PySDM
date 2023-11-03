import numpy as np

from PySDM import Builder, Formulae
from PySDM import products as PySDM_products
from PySDM.backends import CPU
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectro_glacial_sampling import (
    SpectroGlacialSampling,
)

A_VALUE_LARGER_THAN_ONE = 44


def make_particulator(
    *,
    constants,
    n_sd,
    dt,
    initial_temperature,
    singular,
    seed,
    shima_T_fz,
    ABIFM_spec,
    droplet_volume,
    total_particle_number,
    volume,
    thaw=False
):
    attributes = {"volume": np.ones(n_sd) * droplet_volume}

    formulae_ctor_args = {
        "seed": seed,
        "constants": constants,
        "freezing_temperature_spectrum": shima_T_fz,
        "heterogeneous_ice_nucleation_rate": "ABIFM",
        "particle_shape_and_density": "MixedPhaseSpheres",
    }
    formulae = Formulae(**formulae_ctor_args)
    backend = CPU(formulae)

    sampling = SpectroGlacialSampling(
        freezing_temperature_spectrum=formulae.freezing_temperature_spectrum,
        insoluble_surface_spectrum=ABIFM_spec,
    )
    if singular:
        (
            attributes["freezing temperature"],
            _,
            attributes["multiplicity"],
        ) = sampling.sample(backend=backend, n_sd=n_sd)
    else:
        (
            _,
            attributes["immersed surface area"],
            attributes["multiplicity"],
        ) = sampling.sample(backend=backend, n_sd=n_sd)
    attributes["multiplicity"] *= total_particle_number

    env = Box(dt, volume)
    builder = Builder(n_sd, backend, env)

    env["T"] = initial_temperature
    env["RH"] = A_VALUE_LARGER_THAN_ONE
    env["rhod"] = 1.0

    builder.add_dynamic(Freezing(singular=singular, thaw=thaw))

    return builder.build(
        attributes=attributes,
        products=(
            PySDM_products.Time(name="t"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.SpecificIceWaterContent(name="qi"),
        ),
    )
