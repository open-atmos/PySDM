import numpy as np

from PySDM import Particulator, Formulae
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
    thaw=False,
):
    formulae_ctor_args = {
        "seed": seed,
        "constants": constants,
        "freezing_temperature_spectrum": shima_T_fz,
        "heterogeneous_ice_nucleation_rate": "ABIFM",
        "particle_shape_and_density": "MixedPhaseSpheres",
    }
    formulae = Formulae(**formulae_ctor_args)
    backend = CPU(formulae, override_jit_flags={"parallel": False})

    attributes = {
        "signed water mass": np.ones(n_sd) * droplet_volume * formulae.constants.rho_w
    }

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
        immersion_freezing_flag = "singular"
    else:
        (
            _,
            attributes["immersed surface area"],
            attributes["multiplicity"],
        ) = sampling.sample(backend=backend, n_sd=n_sd)
        immersion_freezing_flag = "time-dependent"
    if thaw:
        thaw_flag = "instantaneous"
    else:
        thaw_flag = None
    attributes["multiplicity"] *= total_particle_number

    env = Box(dt, volume, backend=backend)
    env["T"] = initial_temperature
    env["RH"] = A_VALUE_LARGER_THAN_ONE
    env["rhod"] = 1.0

    return Particulator(
        n_sd=n_sd,
        environment=env,
        dynamics=[Freezing(immersion_freezing=immersion_freezing_flag, thaw=thaw_flag)],
        attributes=attributes,
        requested_attributes=("volume",),
        products=(
            PySDM_products.Time(name="t"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.SpecificIceWaterContent(name="qi"),
        ),
    )
