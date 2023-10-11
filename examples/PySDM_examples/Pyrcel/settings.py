from typing import Dict

import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.initialisation.impl.spectrum import Spectrum


@strict
class Settings:
    def __init__(
        self,
        dz: float,
        n_sd_per_mode: tuple,
        aerosol_modes_by_kappa: Dict[float, Spectrum],
        vertical_velocity: float,
        initial_temperature: float,
        initial_pressure: float,
        initial_relative_humidity: float,
        displacement: float,
        formulae: Formulae,
    ):
        self.formulae = formulae
        self.n_sd_per_mode = n_sd_per_mode
        self.aerosol_modes_by_kappa = aerosol_modes_by_kappa

        const = self.formulae.constants
        self.vertical_velocity = vertical_velocity
        self.initial_pressure = initial_pressure
        self.initial_temperature = initial_temperature
        pv0 = (
            initial_relative_humidity
            * formulae.saturation_vapour_pressure.pvs_Celsius(
                initial_temperature - const.T0
            )
        )
        self.initial_vapour_mixing_ratio = const.eps * pv0 / (initial_pressure - pv0)
        self.t_max = displacement / vertical_velocity
        self.timestep = dz / vertical_velocity
        self.output_interval = self.timestep

    @property
    def initial_air_density(self):
        return self.formulae.state_variable_triplet.rho_of_rhod_and_water_vapour_mixing_ratio(
            rhod=self.formulae.trivia.p_d(
                self.initial_pressure, self.initial_vapour_mixing_ratio
            )
            / self.initial_temperature
            / self.formulae.constants.Rd,
            water_vapour_mixing_ratio=self.initial_vapour_mixing_ratio,
        )

    @property
    def nt(self) -> int:
        nt = self.t_max / self.timestep
        nt_int = round(nt)
        np.testing.assert_almost_equal(nt, nt_int)
        return nt_int

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.timestep)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.nt + 1, self.steps_per_output_interval)
