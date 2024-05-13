from typing import Optional

from pystrict import strict
from PySDM import Formulae
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal, Sum

INITIAL_RELATIVE_HUMIDITY = 0.8561
INITIAL_TEMPERATURE = 284.3 * si.K
INITIAL_PRESSURE = 938.5 * si.hPa
INITIAL_ALTITUDE = 600 * si.metres


@strict
class Settings:
    def __init__(self, *, aerosol: str, cloud_type: str, dt: Optional[float] = None):
        self.p0 = INITIAL_PRESSURE
        self.RH0 = INITIAL_RELATIVE_HUMIDITY
        self.T0 = INITIAL_TEMPERATURE
        self.z0 = INITIAL_ALTITUDE
        self.t_end_of_ascent = 1500 * si.s if cloud_type == "Sc" else None
        self.dt = dt or 1 * si.s  # TODO #1266: not found in the paper yet

        self.kappa = 1.28  # Table 1 from Petters & Kreidenweis 2007

        self.formulae = Formulae(
            saturation_vapour_pressure="FlatauWalkoCotton",  # TODO #1266: Bolton
            diffusion_kinetics="GrabowskiEtAl2011",
            diffusion_thermics="GrabowskiEtAl2011",
            constants={
                # values from appendix B
                "MAC": 0.036,
                "HAC": 0.7,
            },
        )

        self.vertical_velocity = {
            # Table 2 in the paper
            "Sc": lambda t: (1 if t < self.t_end_of_ascent else -1) * 0.4 * si.m / si.s,
            "Cu": 2 * si.m / si.s,
        }[cloud_type]

        self.dry_radii_spectrum = {
            # Table 1 in the paper
            "modified polluted": Sum(
                (
                    Lognormal(
                        norm_factor=48 / si.cm**3, m_mode=0.029 * si.um, s_geom=1.36
                    ),
                    Lognormal(
                        norm_factor=114 / si.cm**3, m_mode=0.071 * si.um, s_geom=1.57
                    ),
                )
            ),
            "pristine": Sum(
                (
                    Lognormal(
                        norm_factor=125 / si.cm**3, m_mode=0.011 * si.um, s_geom=1.2
                    ),
                    Lognormal(
                        norm_factor=65 / si.cm**3, m_mode=0.06 * si.um, s_geom=1.7
                    ),
                )
            ),
        }[aerosol]
