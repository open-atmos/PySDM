from pystrict import strict

from PySDM import Formulae
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal


@strict
class Settings:
    def __init__(self, *, n_sd: int, timestep: float):
        self.n_sd = n_sd
        self.timestep = timestep

        self.initial_total_pressure = 1000 * si.hPa  # note: not given in the paper

        # parameters from the paper
        self.formulae = Formulae(
            constants={"bulk_phase_partitioning_exponent": 0.1},
            bulk_phase_partitioning="KaulEtAl2015",
        )
        self.initial_water_vapour_mixing_ratio = 1.5 * si.g / si.kg
        self.parcel_linear_extent = 100 * si.m
        self.updraft = 0.5 * si.m / si.s

        thd_0 = self.formulae.state_variable_triplet.th_dry(
            th_std=269 * si.K,
            water_vapour_mixing_ratio=self.initial_water_vapour_mixing_ratio,
        )
        rhod_0 = self.formulae.state_variable_triplet.rho_d(
            p=self.initial_total_pressure,
            water_vapour_mixing_ratio=self.initial_water_vapour_mixing_ratio,
            theta_std=thd_0,
        )

        self.mass_of_dry_air = rhod_0 * self.formulae.trivia.volume(
            radius=self.parcel_linear_extent
        )
        self.soluble_aerosol = Lognormal(
            norm_factor=200
            / si.mg
            * self.mass_of_dry_air,  # note: assuming per mg of dry air
            m_mode=75 * si.nm,
            s_geom=1.6,
        )
        self.kappa = 0.6
        self.initial_temperature = self.formulae.state_variable_triplet.T(
            rhod=rhod_0, thd=thd_0
        )
