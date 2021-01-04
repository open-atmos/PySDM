"""
Created at 29.11.2019
"""

from PySDM.physics.constants import si
from PySDM.physics import constants as const, formulae as phys
from PySDM.dynamics import condensation
import numpy as np


class Settings:
    def __init__(self, w_avg, N_STP, r_dry, mass_of_dry_air):
        self.q0 = const.eps / (self.p0 / self.RH0 / phys.pvs(self.T0) - 1)
        self.w_avg = w_avg
        self.r_dry = r_dry
        self.N_STP = N_STP
        self.n_in_dv = N_STP / const.rho_STP * mass_of_dry_air
        self.mass_of_dry_air = mass_of_dry_air

    @property
    def dt_max(self):
        t_total = 2 * Settings.z_half / self.w_avg
        result = t_total / Settings.n_output
        if result < 1 * si.centimetre / si.second:
            result /= 100  # TODO
        return result

    n_output = 500

    rtol_x = condensation.default_rtol_x
    rtol_thd = condensation.default_rtol_thd
    coord = 'volume logarithm'

    p0 = 1000 * si.hectopascals
    RH0 = .98
    kappa = .2
    T0 = 300 * si.kelvin
    z_half = 150 * si.metres

    def w(self, t):
        return self.w_avg * np.pi / 2 * np.sin(np.pi * t / self.z_half * self.w_avg)


w_avgs = (
    100 * si.centimetre / si.second,
    # 50 * si.centimetre / si.second,
    .2 * si.centimetre / si.second
)

N_STPs = (
    50 / si.centimetre ** 3,
    500 / si.centimetre ** 3
)

r_drys = (
    .1 * si.micrometre,
    .05 * si.micrometre
)

setups = []
for w_i in range(len(w_avgs)):
    for N_i in range(len(N_STPs)):
        for rd_i in range(len(r_drys)):
            if not rd_i == N_i == 1:
                setups.append(Settings(
                    w_avg=w_avgs[w_i],
                    N_STP=N_STPs[N_i],
                    r_dry=r_drys[rd_i],
                    mass_of_dry_air=1000 * si.kilogram
                ))
