"""
Created at 29.11.2019

@author: Michael Olesik
@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.backends.default import Default
from PySDM.simulation.physics.constants import si
from PySDM.simulation.physics import constants as const
from PySDM.simulation.physics import formulae as phys
import numpy as np


class Setup:
    def __init__(self, w_avg, N, r_dry):
        self.q0 = const.eps / (self.p0 / self.RH0 / phys.pvs(self.T0) - 1)
        rho0 = self.p0 / phys.R(self.q0) / self.T0

        self.w_avg = w_avg
        self.r_dry = r_dry
        self.N = N
        self.n_per_mass = N / rho0 * self.mass

    backend = Default
    mass = 1000 * si.kilogram  # TODO: it should not matter, but it does !!!!!!!!!!!!1
    n_steps = 500

    p0 = 1000 * si.hectopascals
    RH0 = .98
    kappa = .2
    T0 = 300 * si.kelvin
    z_half = 150 * si.metres

    def w(self, t):
        return self.w_avg * np.pi / 2 * np.sin(np.pi * t / self.z_half * self.w_avg)


w_avgs = (
    100 * si.centimetre / si.second,
    50 * si.centimetre / si.second,
    .2 * si.centimetre / si.second
)

Ns = (
    50 / si.centimetre ** 3,
    500 / si.centimetre ** 3
)

rds = (
    .1 * si.micrometre,
    .05 * si.micrometre
)

setups = []
for w_i in range(len(w_avgs)):
    for N_i in range(len(Ns)):
        for rd_i in range(len(rds)):
            if not rd_i == N_i == 1:
                setups.append(Setup(
                    w_avg=w_avgs[w_i],
                    N=Ns[N_i],
                    r_dry=rds[rd_i]
                ))
