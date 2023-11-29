import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.dynamics import condensation
from PySDM.physics.constants import si


@strict
class Settings:
    def __init__(
        self,
        w_avg: float,
        N_STP: float,
        r_dry: float,
        mass_of_dry_air: float,
        coord: str = "VolumeLogarithm",
    ):
        self.formulae = Formulae(
            saturation_vapour_pressure="AugustRocheMagnus",
            condensation_coordinate=coord,
        )
        const = self.formulae.constants
        self.p0 = 1000 * si.hectopascals
        self.RH0 = 0.98
        self.kappa = 0.2  # TODO #441
        self.T0 = 300 * si.kelvin
        self.z_half = 150 * si.metres

        pvs = self.formulae.saturation_vapour_pressure.pvs_Celsius(self.T0 - const.T0)
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.p0 / self.RH0 / pvs - 1
        )
        self.w_avg = w_avg
        self.r_dry = r_dry
        self.N_STP = N_STP
        self.n_in_dv = N_STP / const.rho_STP * mass_of_dry_air
        self.mass_of_dry_air = mass_of_dry_air
        self.n_output = 500

        self.rtol_x = condensation.DEFAULTS.rtol_x
        self.rtol_thd = condensation.DEFAULTS.rtol_thd
        self.coord = "volume logarithm"
        self.dt_cond_range = condensation.DEFAULTS.cond_range

    @property
    def dt_max(self):
        t_total = 2 * self.z_half / self.w_avg
        result = t_total / self.n_output
        if result < 1 * si.centimetre / si.second:
            result /= 100  # TODO #411
        return result

    def w(self, t):
        return self.w_avg * np.pi / 2 * np.sin(np.pi * t / self.z_half * self.w_avg)


w_avgs = (
    100 * si.centimetre / si.second,
    # 50 * si.centimetre / si.second,
    0.2 * si.centimetre / si.second,
)

N_STPs = (50 / si.centimetre**3, 500 / si.centimetre**3)

r_drys = (0.1 * si.micrometre, 0.05 * si.micrometre)

setups = []
for w_i, _w_avg in enumerate(w_avgs):
    for N_i, _N_STP in enumerate(N_STPs):
        for rd_i, _r_dry in enumerate(r_drys):
            if not rd_i == N_i == 1:
                setups.append(
                    Settings(
                        w_avg=_w_avg,
                        N_STP=_N_STP,
                        r_dry=_r_dry,
                        mass_of_dry_air=1000 * si.kilogram,
                    )
                )
