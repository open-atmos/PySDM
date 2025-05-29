import numpy as np
from pystrict import strict
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si
from PySDM import Formulae


@strict
class Settings:
    def __init__(
        self,
        *,
        super_droplet_injection_rate: callable,
        n_sd_initial: int,
        n_sd_seeding: int,
        rain_water_radius_threshold: float,
        formulae: Formulae,
        enable_collisions: bool = True,
    ):
        self.enable_collisions = enable_collisions
        self.formulae = formulae
        self.n_sd_initial = n_sd_initial
        self.n_sd_seeding = n_sd_seeding
        self.rain_water_radius_threshold = rain_water_radius_threshold

        self.t_max = 25 * si.min
        self.w_max = 3 * si.m / si.s
        self.w_min = 0.025 * si.m / si.s

        self.timestep = 15 * si.s
        self.mass_of_dry_air = 666 * si.kg

        self.updraft = (
            lambda t: self.w_min
            + (self.w_max - self.w_min)
            * np.maximum(0, np.sin(t / self.t_max * 2 * np.pi)) ** 2
        )
        self.initial_water_vapour_mixing_ratio = 666 / 30 * si.g / si.kg
        self.initial_total_pressure = 1000 * si.hPa
        self.initial_temperature = 300 * si.K
        self.initial_aerosol_kappa = 0.5
        self.initial_aerosol_dry_radii = Lognormal(
            norm_factor=200 / si.mg * self.mass_of_dry_air,
            m_mode=75 * si.nm,
            s_geom=1.6,
        )
        self.super_droplet_injection_rate = super_droplet_injection_rate

        r_dry, n_in_dv = ConstantMultiplicity(
            Lognormal(
                norm_factor=10 / si.mg * self.mass_of_dry_air,
                m_mode=1 * si.um,
                s_geom=1.1,
            )
        ).sample(
            n_sd=n_sd_seeding
        )  # TODO #1387: does not to be the same?
        v_dry = self.formulae.trivia.volume(radius=r_dry)
        self.seeded_particle_multiplicity = n_in_dv
        self.seeded_particle_extensive_attributes = {
            "signed water mass": [0.0001 * si.ng] * n_sd_seeding,
            "dry volume": v_dry,
            "kappa times dry volume": 0.8 * v_dry,
        }
