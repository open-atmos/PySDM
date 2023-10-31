from typing import Optional

from pystrict import strict

from PySDM import Formulae
from PySDM.physics import constants as const
from PySDM.physics import si


@strict
class Settings:
    def __init__(
        self,
        *,
        formulae: Optional[Formulae],
        ccn_sampling_n: int = 11,
        in_sampling_n: int = 20,
        initial_temperature: float,
        timestep: float
    ):
        self.ccn_sampling_n = ccn_sampling_n
        self.in_sampling_n = in_sampling_n

        self.timestep = timestep
        self.initial_temperature = initial_temperature
        self.formulae = formulae

        self.initial_relative_humidity = 0.985
        self.vertical_velocity = 20 * si.cm / si.s
        self.displacement = 300 * si.m
        self.kappa = 0.53  # ammonium sulfate (Tab. 1 in P&K07)
        self.mass_of_dry_air = 1e3 * si.kg

        # note: 2000 um in the paper... but it gives 0 concentrations
        self.ccn_dry_diameter_range = (10 * si.nm, 353 * si.nm)

    @property
    def p0(self):
        return 1000 * si.hPa

    @property
    def pv0(self):
        pvs = self.formulae.saturation_vapour_pressure.pvs_Celsius(self.T0 - const.T0)
        return self.initial_relative_humidity * pvs

    @property
    def initial_water_vapour_mixing_ratio(self):
        pv0 = self.pv0
        return self.formulae.constants.eps * pv0 / (self.p0 - pv0)

    @property
    def T0(self):
        return self.initial_temperature

    @property
    def rhod0(self):
        rho_v = self.pv0 / self.formulae.constants.Rv / self.T0
        return rho_v / self.initial_water_vapour_mixing_ratio
