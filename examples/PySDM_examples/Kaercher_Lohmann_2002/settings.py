import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.physics.constants import si
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling import spectral_sampling


@strict
class Settings:
    def __init__(
        self,
        n_sd: int,
        w_updraft: float,
        T0: float,
        N_solution_droplet: float,
        r_mean: float,
        kappa: float,
    ):
        # print( n_sd, w_updraft, T0, N_solution_droplet, r_mean )

        self.formulae = Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            homogeneous_ice_nucleation_rate="Constant",
            constants={"J_HOM": 1e15},
        )
        formulae = self.formulae
        const = formulae.constants

        mass_of_dry_air = 1000 * si.kilogram

        self.w_updraft = w_updraft
        self.mass_of_dry_air = mass_of_dry_air
        self.p0 = 220 * si.hectopascals
        self.T0 = T0
        self.RHi0 = 1.0
        pvs_i = formulae.saturation_vapour_pressure.pvs_ice(self.T0)
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.p0 / self.RHi0 / pvs_i - 1
        )

        dry_air_density = (
            formulae.trivia.p_d(self.p0, self.initial_water_vapour_mixing_ratio)
            / formulae.constants.Rd
            / self.T0
        )

        # N_solution_droplet = 1e4 / si.centimetre**3
        spectrum = Lognormal(
            norm_factor=N_solution_droplet / dry_air_density, m_mode=r_mean, s_geom=1.6
        )

        self.n_sd = n_sd
        self.r_dry, self.specific_concentration = spectral_sampling.Logarithmic(
            spectrum
        ).sample(n_sd)
        self.kappa = kappa  # 0.64

        self.r_mean = r_mean
        self.N_solution_drople = N_solution_droplet
        self.n_in_dv = N_solution_droplet / const.rho_STP * mass_of_dry_air

        self.t_duration = 3600.0  # 60. #5400 # total duration of simulation
        self.dt = 1.0
        self.n_output = 60  # 10 # number of output steps


n_sds = (100,)

w_updrafts = (50 * si.centimetre / si.second,)

T_starts = (220 * si.kelvin,)

N_solution_droplets = (2500 / si.centimetre**3,)

r_means = (0.055 * si.micrometre,)

# kappas=( 0.3, 0.64, 0.9, 1.2 )
kappas = (0.64,)

setups = []
for n_sd in n_sds:
    for w_updraft in w_updrafts:
        for T0 in T_starts:
            for N_solution_droplet in N_solution_droplets:
                for r_mean in r_means:
                    for kappa in kappas:
                        setups.append(
                            Settings(
                                n_sd=n_sd,
                                w_updraft=w_updraft,
                                T0=T0,
                                N_solution_droplet=N_solution_droplet,
                                r_mean=r_mean,
                                kappa=kappa,
                            )
                        )
