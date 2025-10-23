"""commons for homogeneous freezing notebooks"""

import time
from PySDM_examples.Luettmer_homogeneous_freezing.settings import Settings
from PySDM_examples.Luettmer_homogeneous_freezing.simulation import Simulation
from PySDM import Formulae
from PySDM.physics.constants import si
from PySDM.backends import CPU

formulae = Formulae(
    particle_shape_and_density="MixedPhaseSpheres",
)


def run_simulations(setting):

    simulation = {
        "settings": setting,
        "ensemble_member_outputs": [],
    }
    for _ in range(setting["number_of_ensemble_runs"]):
        model_setup = Settings(**simulation["settings"])
        model_setup.formulae.seed += 1
        model = Simulation(model_setup)
        simulation["ensemble_member_outputs"].append(model.run())

    return simulation


def hom_pure_droplet_freezing_backend():
    backends = {
        "threshold": CPU(
            formulae=Formulae(
                particle_shape_and_density="MixedPhaseSpheres",
                homogeneous_ice_nucleation_rate="Null",
                saturation_vapour_pressure="MurphyKoop2005",
                seed=time.time_ns(),
            )
        ),
        "KoopMurray2016": CPU(
            formulae=Formulae(
                particle_shape_and_density="MixedPhaseSpheres",
                homogeneous_ice_nucleation_rate="KoopMurray2016",
                saturation_vapour_pressure="MurphyKoop2005",
                seed=time.time_ns(),
            )
        ),
        "Spichtinger2023": CPU(
            formulae=Formulae(
                particle_shape_and_density="MixedPhaseSpheres",
                homogeneous_ice_nucleation_rate="Koop_Correction",
                saturation_vapour_pressure="MurphyKoop2005",
                seed=time.time_ns(),
            )
        ),
        "Koop2000": CPU(
            formulae=Formulae(
                particle_shape_and_density="MixedPhaseSpheres",
                homogeneous_ice_nucleation_rate="Koop2000",
                saturation_vapour_pressure="MurphyKoop2005",
                seed=time.time_ns(),
            )
        ),
    }
    return backends


def hom_pure_droplet_freezing_standard_setup():
    standard = {
        "n_sd": int(1e3),
        "w_updraft": 1.0 * si.meter / si.second,
        "T0": formulae.trivia.C2K(-25),
        "dz": 0.1 * si.meter,
        "N_dv_droplet_distribution": 750 / si.cm**3,
        "r_mean_droplet_distribution": 15 * si.nanometer,
        "type_droplet_distribution": "monodisperse",
        "RH_0": 0.995,
        "p0": 500 * si.hectopascals,
        "condensation_enable": True,
        "deposition_enable": True,
        "deposition_adaptive": True,
        "number_of_ensemble_runs": 1,
    }
    return standard
