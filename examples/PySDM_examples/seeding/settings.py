import numpy as np
from pystrict import strict
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si


@strict
class Settings:
    def __init__(self, *, super_droplet_injection_rate: callable):
        self.n_sd_initial = 100
        self.n_sd_seeding = 100
        self.t_max = 20 * si.min
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
        self.seeded_particle_multiplicity = 1
        self.seeded_particle_extensive_attributes = {
            "water mass": 0.001 * si.ng,
            "dry volume": 0.0001 * si.ng,
            "kappa times dry volume": 0.8 * 0.0001 * si.ng,
        }
