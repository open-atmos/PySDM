import time

import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.physics.constants import si
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling import spectral_sampling

@strict
class settings:
    def __init__(
        self,
        *,
        n_sd: int,
        w_updraft: float,
        T0: float,
        RHi_0: float=1.4,
        N_dv_solution_droplet: float,
        r_mean_solution_droplet: float,
        sigma_solution_droplet: float,
        kappa: float=0.64,
        rate: str="Koop2000",
        dt: float=0.1,
        linear_sampling: bool=False,
        lower_limit: float=None,
    ):

        self.n_sd = n_sd
        self.w_updraft = w_updraft
        self.r_mean_solution_droplet = r_mean_solution_droplet
        self.N_dv_solution_droplet = N_dv_solution_droplet
        self.sigma_solution_droplet = sigma_solution_droplet
        self.rate = rate

        self.mass_of_dry_air = 1000 * si.kilogram
        self.initial_pressure = 220 * si.hectopascals
        self.initial_ice_supersaturation = RHi_0
        self.kappa = kappa
        self.initial_temperature  = T0
        print( "RHi0", self.initial_ice_supersaturation )
        print("T0",   self.initial_temperature)
        print( "w_updraft",  self.w_updraft )
        self.formulae = Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            homogeneous_ice_nucleation_rate=rate,
            constants={"J_HOM": 1.e15},
            seed=time.time_ns()
        )
        const = self.formulae.constants
        pvs_i = self.formulae.saturation_vapour_pressure.pvs_ice(self.initial_temperature)
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.initial_pressure / self.initial_ice_supersaturation / pvs_i - 1
        )
        dry_air_density =  (self.formulae.trivia.p_d(self.initial_pressure, self.initial_water_vapour_mixing_ratio )
                            / self.initial_temperature
                            / const.Rd )

        spectrum = Lognormal(norm_factor=N_dv_solution_droplet / dry_air_density,
                             m_mode=r_mean_solution_droplet,
                             s_geom=sigma_solution_droplet)



        upper_limit = 4e-6

        if linear_sampling:
            self.r_dry, self.specific_concentration = (
                spectral_sampling.Linear(spectrum).sample(n_sd))
        else:
            if lower_limit is None:
                self.r_dry, self.specific_concentration = (
                    spectral_sampling.Logarithmic(spectrum,
                                                  ).sample(n_sd))
            else:
                self.r_dry, self.specific_concentration = (
                    spectral_sampling.Logarithmic(spectrum,
                                                  size_range=(lower_limit,upper_limit),
                                                  error_threshold=0.95,
                ).sample(n_sd))
                test_for_zero_n = self.specific_concentration == 0.
                if( any(test_for_zero_n) ):
                    last_non_zero = np.nonzero(self.specific_concentration)[0][-1]
                    upper_limit = self.specific_concentration[last_non_zero]
                    self.r_dry, self.specific_concentration = (
                        spectral_sampling.Logarithmic(spectrum,
                                                      size_range=(lower_limit, upper_limit),
                                                      error_threshold=0.95,
                                                      ).sample(n_sd))

        self.t_duration = 3600 #3600 * 1.5 # total duration of simulation
        self.dt         = dt
        self.n_output   = 3600 #100 # number of output steps

