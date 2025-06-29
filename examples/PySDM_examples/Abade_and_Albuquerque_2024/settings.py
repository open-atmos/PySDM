from pystrict import strict

from PySDM.backends import CPU
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal


@strict
class Settings:
    def __init__(
        self,
        *,
        backend: CPU,
        n_sd: int,
        timestep: float,
        # default values correspond to paper settings
        singular: bool = True,
        enable_immersion_freezing: bool = True,
        enable_vapour_deposition_on_ice: bool = True,
        inp_frac: float = 0.1,
        kappa: float = 0.6,
        updraft: float = 0.5 * si.m / si.s,
    ):
        self.backend = backend
        self.n_sd = n_sd
        self.timestep = timestep
        self.enable_immersion_freezing = enable_immersion_freezing
        self.enable_vapour_deposition_on_ice = enable_vapour_deposition_on_ice
        self.singular = singular
        self.initial_total_pressure = 1000 * si.hPa  # note: not given in the paper

        self.initial_water_vapour_mixing_ratio = 1.5 * si.g / si.kg
        self.parcel_linear_extent = 100 * si.m
        self.updraft = updraft
        self.freezing_inp_frac = inp_frac
        self.freezing_inp_dry_radius = 0.5 * si.um

        thd_0 = backend.formulae.state_variable_triplet.th_dry(
            th_std=269 * si.K,
            water_vapour_mixing_ratio=self.initial_water_vapour_mixing_ratio,
        )
        rhod_0 = backend.formulae.state_variable_triplet.rho_d(
            p=self.initial_total_pressure,
            water_vapour_mixing_ratio=self.initial_water_vapour_mixing_ratio,
            theta_std=thd_0,
        )

        self.mass_of_dry_air = rhod_0 * backend.formulae.trivia.volume(
            radius=self.parcel_linear_extent
        )
        self.soluble_aerosol = Lognormal(
            norm_factor=200
            / si.mg
            * self.mass_of_dry_air,  # note: assuming per mg of dry air
            m_mode=75 * si.nm,
            s_geom=1.6,
        )
        self.kappa = kappa
        self.initial_temperature = backend.formulae.state_variable_triplet.T(
            rhod=rhod_0, thd=thd_0
        )
