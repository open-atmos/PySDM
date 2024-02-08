from PySDM import Formulae
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal, Sum
from pystrict import strict


@strict
class Settings:
    def __init__(self, *, aerosol: str, cloud_type: str):
        self.p0 = 938.5 * si.hPa
        self.RH0 = 0.8561
        self.T0 = 284.3 * si.K

        self.kappa = 1.28  # Table 1 from Petters & Kreidenweis 2007

        self.formulae = Formulae(
            saturation_vapour_pressure="FlatauWalkoCotton",  # TODO: Bolton
            constants={
                # values from appendix B
                "MAC": 0.036,
                "HAC": 0.7,
            },
        )

        self.vertical_velocity = {
            # Table 2 in the paper
            "Sc": lambda t: (1 if t < 1500 * si.s else -1) * 0.4 * si.m / si.s,
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
