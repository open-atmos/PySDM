import time

import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.physics.constants import si
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.dynamics.collisions.collision_kernels import Geometric


class Settings:
    def __init__(
        self,
        *,
        hom_freezing: str,
        n_sd: int,
        w_updraft: float,
        T0: float,
        dz: float,
        N_dv_droplet_distribution: float,
        r_mean_droplet_distribution: float,
        sigma_droplet_distribution: float = None,
        type_droplet_distribution: str,
        p0: float = 200 * si.hectopascals,
        RH_0: float = 1.0,
        kappa: float = 0.64,
        condensation_enable: bool = True,
        deposition_enable: bool = True,
        coalescence_enable: bool = False,
        deposition_adaptive: bool = True,
        n_output: int = 30,
        backend=None,
        scipy_solver=False,
        number_of_ensemble_runs: None,
    ):
        self.backend = backend
        print(
            "Setting up simulation for "
            + hom_freezing
            + " with wpdraft="
            + str(w_updraft)
            + " and n_sd="
            + str(n_sd)
            + " and n_dv="
            + str(N_dv_droplet_distribution)
        )
        self.n_sd = n_sd
        self.w_updraft = w_updraft
        self.N_dv_droplet_distribution = N_dv_droplet_distribution
        self.r_mean_droplet_distribution = r_mean_droplet_distribution
        self.sigma_droplet_distribution = sigma_droplet_distribution
        self.type_droplet_distribution = type_droplet_distribution

        self.mass_of_dry_air = 1000 * si.kilogram
        self.initial_pressure = p0
        self.initial_water_supersaturation = RH_0
        self.kappa = kappa
        self.initial_temperature = T0

        self.condensation_enable = condensation_enable
        self.coalescence_enable = coalescence_enable
        self.deposition_enable = deposition_enable
        self.deposition_adaptive = deposition_adaptive
        self.scipy_solver = scipy_solver
        self.collision_kernel = Geometric()

        if hom_freezing == "threshold":
            self.hom_freezing_type = "threshold"
        else:
            self.hom_freezing_type = "time-dependent"

        self.formulae = backend.formulae

        const = self.formulae.constants
        pvs_w = self.formulae.saturation_vapour_pressure.pvs_water(
            self.initial_temperature
        )
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.initial_pressure / self.initial_water_supersaturation / pvs_w - 1
        )

        dry_air_density = (
            self.formulae.trivia.p_d(
                self.initial_pressure, self.initial_water_vapour_mixing_ratio
            )
            / self.initial_temperature
            / const.Rd
        )

        if self.type_droplet_distribution == ("monodisperse"):
            self.r_wet = np.ones(self.n_sd) * r_mean_droplet_distribution
            self.specific_concentration = (
                np.ones(self.n_sd)
                * N_dv_droplet_distribution
                / self.n_sd
                / dry_air_density
            )
            if coalescence_enable:  # do lucky droplet method
                v_small = self.formulae.trivia.volume(
                    radius=r_mean_droplet_distribution
                )
                self.r_wet[0 : int(self.n_sd * 0.1)] = self.formulae.trivia.radius(
                    volume=2 * v_small
                )
        elif self.type_droplet_distribution == ("lognormal"):
            spectrum = Lognormal(
                norm_factor=N_dv_droplet_distribution / dry_air_density,
                m_mode=r_mean_droplet_distribution,
                s_geom=sigma_droplet_distribution,
            )
            self.r_wet, self.specific_concentration = spectral_sampling.Linear(
                spectrum
            ).sample(n_sd)

        self.dz = dz
        self.t_max_duration = 10000
        self.dt = dz / self.w_updraft
        self.n_output = n_output
