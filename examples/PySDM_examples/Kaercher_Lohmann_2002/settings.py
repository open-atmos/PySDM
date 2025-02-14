import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.physics.constants import si



@strict
class Settings:
    def __init__(
        self,
        w_updraft: float,
        T0: float,
        N_solution_droplet: float,
        r_solution_droplet: float,
    ):
        print( w_updraft, T0, N_solution_droplet, r_solution_droplet )
        mass_of_dry_air = 1000 * si.kilogram
        self.formulae = Formulae(
 #           saturation_vapour_pressure="AugustRocheMagnus",
        )
        const = self.formulae.constants
        self.mass_of_dry_air = mass_of_dry_air 
        self.p0 = 220 * si.hectopascals
        self.RHi0 = 1.
        self.kappa = 0.64
        self.T0 = T0
    
        pvs_i = self.formulae.saturation_vapour_pressure.pvs_ice(self.T0)
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.p0 / self.RHi0 / pvs_i - 1
        )
        self.w_updraft = w_updraft
        self.r_solution_droplet = r_solution_droplet
        self.N_solution_drople = N_solution_droplet
        self.n_in_dv = N_solution_droplet / const.rho_STP * mass_of_dry_air


        self.t_duration = 5400 # total duration of simulation
        self.dt         = 1. 
        self.n_output = 10 # number of output steps


w_updrafts = (
    10 * si.centimetre / si.second,
)
        
T_starts = ( 220 * si.kelvin, )

N_solution_droplets = ( 2500 / si.centimetre**3, )

r_solution_droplets = ( 0.0555 * si.micrometre, )

setups = []
for w_updraft in w_updrafts:
    for T0 in T_starts:
        for N_solution_droplet in N_solution_droplets:
            for r_solution_droplet in r_solution_droplets:
                setups.append(
                    Settings(
                        w_updraft=w_updraft,
                        T0=T0,
                        N_solution_droplet=N_solution_droplet,
                        r_solution_droplet=r_solution_droplet,
                    )
                )
