from pystrict import strict

from PySDM import Formulae
from PySDM.physics.constants import si
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling import spectral_sampling


@strict
class Settings:
    def __init__(
        self,
        *,
        n_sd: int,
        w_updraft: float,
        T0: float,
        seed: int,
        dt: float,
    ):

        self.n_sd = n_sd
        self.w_updraft = w_updraft

        self.N_dv_solution_droplet = 2500 / si.centimetre**3
        self.r_mean_solution_droplet = 0.055 * si.micrometre
        self.sigma_solution_droplet = 1.6

        self.mass_of_dry_air = 1000 * si.kilogram
        self.initial_pressure = 200 * si.hectopascals
        self.initial_ice_supersaturation = 1.0
        self.kappa = 0.64
        self.initial_temperature = T0

        self.formulae = Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            homogeneous_ice_nucleation_rate="Koop_Correction",
            seed=seed,
        )
        const = self.formulae.constants
        pvs_i = self.formulae.saturation_vapour_pressure.pvs_ice(
            self.initial_temperature
        )
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.initial_pressure / self.initial_ice_supersaturation / pvs_i - 1
        )
        dry_air_density = (
            self.formulae.trivia.p_d(
                self.initial_pressure, self.initial_water_vapour_mixing_ratio
            )
            / self.initial_temperature
            / const.Rd
        )

        spectrum = Lognormal(
            norm_factor=self.N_dv_solution_droplet / dry_air_density,
            m_mode=self.r_mean_solution_droplet,
            s_geom=self.sigma_solution_droplet,
        )

        self.r_dry, self.specific_concentration = spectral_sampling.Linear(
            spectrum
        ).sample(n_sd)

        self.t_duration = 7200
        self.dt = dt
        self.n_output = int(self.t_duration / 100)
