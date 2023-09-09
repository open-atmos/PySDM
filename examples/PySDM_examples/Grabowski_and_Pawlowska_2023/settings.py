import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.dynamics import condensation
from PySDM.initialisation.spectra import Lognormal, Sum
from PySDM.physics import si

condensation_tolerance = condensation.DEFAULTS.rtol_thd / 100


@strict
class Settings:
    def __init__(
        self,
        *,
        aerosol: str,
        vertical_velocity: float,
        dt: float,
        n_sd: int,
        initial_temperature: float = 283 * si.K,
        initial_pressure: float = 900 * si.mbar,
        initial_relative_humidity: float = 0.97,
        displacement: float = 1000 * si.m,
        mass_accommodation_coefficient: float = 0.3,
        rtol_thd: float = condensation_tolerance,
        rtol_x: float = condensation_tolerance
    ):
        self.formulae = Formulae(constants={"MAC": mass_accommodation_coefficient})
        self.n_sd = n_sd
        self.aerosol_modes_by_kappa = {
            "pristine": {
                1.28: Sum(
                    (
                        Lognormal(
                            norm_factor=125 / si.cm**3, m_mode=11 * si.nm, s_geom=1.2
                        ),
                        Lognormal(
                            norm_factor=65 / si.cm**3, m_mode=60 * si.nm, s_geom=1.7
                        ),
                    )
                )
            },
            "polluted": {
                1.28: Sum(
                    (
                        Lognormal(
                            norm_factor=160 / si.cm**3, m_mode=29 * si.nm, s_geom=1.36
                        ),
                        Lognormal(
                            norm_factor=380 / si.cm**3, m_mode=71 * si.nm, s_geom=1.57
                        ),
                    )
                )
            },
        }[aerosol]

        const = self.formulae.constants
        self.vertical_velocity = vertical_velocity
        self.initial_pressure = initial_pressure
        self.initial_temperature = initial_temperature
        pv0 = (
            initial_relative_humidity
            * self.formulae.saturation_vapour_pressure.pvs_Celsius(
                initial_temperature - const.T0
            )
        )
        self.initial_vapour_mixing_ratio = const.eps * pv0 / (initial_pressure - pv0)
        self.t_max = displacement / vertical_velocity
        self.timestep = dt
        self.output_interval = self.timestep
        self.rtol_thd = rtol_thd
        self.rtol_x = rtol_x

    @property
    def initial_air_density(self):
        const = self.formulae.constants
        dry_air_density = (
            self.formulae.trivia.p_d(
                self.initial_pressure, self.initial_vapour_mixing_ratio
            )
            / self.initial_temperature
            / const.Rd
        )
        return dry_air_density * (1 + self.initial_vapour_mixing_ratio)

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
