from PySDM import Formulae
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal, Sum


class Settings:
    def __init__(self):
        self.dt = 0.25 * si.s
        self.mass_of_dry_air = 1000 * si.kg
        self.p0 = 1000 * si.hPa
        self.RH0 = 0.99 * si.dimensionless
        self.T0 = 280 * si.K
        self.w = 0.25 * si.m / si.s

        self.initial_air_density = 1.245 * si.kg / si.m**3

        self.output_interval = 4
        self.output_points = 250
        self.n_sd = 1024

        self.formulae = Formulae()

        self.cloud_range = (1 * si.um, 25 * si.um)

        self.kappa_sea_salt = 1.28 * si.dimensionless
        self.kappa_sulphate = 0.61 * si.dimensionless

        self.sea_salt_concentration = (
            51.1 / si.cm**3 + 2.21 / si.cm**3 + 1e-5 / si.cm**3
        )
        self.sulphate_concentration = 100 / si.cm**3
        self.total_aerosol_concentration = (
            self.sea_salt_concentration + self.sulphate_concentration
        )

        sea_salt_spectrum = Sum(
            (
                Lognormal(
                    norm_factor=51.1 / si.cm**3,
                    m_mode=0.10 * si.um,
                    s_geom=1.90,
                ),
                Lognormal(
                    norm_factor=2.21 / si.cm**3,
                    m_mode=1.00 * si.um,
                    s_geom=2.00,
                ),
                Lognormal(
                    norm_factor=1e-5 / si.cm**3,
                    m_mode=6.00 * si.um,
                    s_geom=3.00,
                ),
            )
        )

        sulphate_spectrum = Lognormal(
            norm_factor=100 / si.cm**3,
            m_mode=0.08 * si.um,
            s_geom=1.45,
        )

        self.aerosol_modes_by_kappa = {
            self.kappa_sea_salt: sea_salt_spectrum,
            self.kappa_sulphate: sulphate_spectrum,
        }

        self.aerosol_mode_names = (
            "sea salt",
            "sulphate",
        )
