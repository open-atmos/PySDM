import time

import numpy as np
from pystrict import strict


from PySDM.physics.constants import si
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling import spectral_sampling
from tests.unit_tests.backends.test_oxidation import formulae


class Settings:
    def __init__(
        self,
        *,
        formulae: formulae,
        n_sd: int,
        w_updraft: float,
        T0: float,
        dt: float,
        N_dv_droplet_distribution: float,
        r_mean_droplet_distribution: float,
        sigma_droplet_distribution: float = None,
        type_droplet_distribution: str = "monodisperse",
        hom_freezing: str = "threshold",
        p0: float =  200 * si.hectopascals,
        RH_0: float=1.0,
        kappa: float=0.64,
        condensation_enable = True,
        deposition_enable = True,
    ):

        self.n_sd = n_sd
        self.w_updraft = w_updraft
        self.N_dv_droplet_distribution = N_dv_droplet_distribution
        self.r_mean_droplet_distribution = r_mean_droplet_distribution
        self.sigma_droplet_distribution = sigma_droplet_distribution
        self.type_droplet_distribution = type_droplet_distribution

        self.mass_of_dry_air = 1000 * si.kilogram
        self.initial_pressure = p0
        self.initial_water_supersaturation = RH_0
        # self.initial_ice_supersaturation = RHi_0
        self.kappa = kappa
        self.initial_temperature  = T0

        self.condensation_enable = condensation_enable
        self.deposition_enable = deposition_enable
        self.hom_freezing = hom_freezing

        self.formulae = formulae


        const = self.formulae.constants
        # pvs_i = self.formulae.saturation_vapour_pressure.pvs_ice(self.initial_temperature)
        pvs_w = self.formulae.saturation_vapour_pressure.pvs_water(self.initial_temperature)
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.initial_pressure / self.initial_water_supersaturation / pvs_w - 1
        )

        dry_air_density =  (self.formulae.trivia.p_d(self.initial_pressure, self.initial_water_vapour_mixing_ratio )
                            / self.initial_temperature
                            / const.Rd )

        if self.type_droplet_distribution == ("monodisperse"):
            self.droplet_radius = np.ones( self.n_sd ) * r_mean_droplet_distribution
            self.specific_concentration = np.ones( self.n_sd ) * N_dv_droplet_distribution / self.n_sd / dry_air_density

        elif self.type_droplet_distribution == ("lognormal"):
            spectrum = Lognormal(norm_factor=N_dv_droplet_distribution / dry_air_density,
                                 m_mode=r_mean_droplet_distribution,
                                 s_geom=sigma_droplet_distribution)

            self.r_dry, self.specific_concentration = (spectral_sampling.Linear(spectrum).sample(n_sd))


        self.t_max_duration = 7200 #3600 * 1.5 # total duration of simulation
        self.dt         = dt
        self.n_output   = 10 #int(self.t_duration / 100) #100 # number of output steps