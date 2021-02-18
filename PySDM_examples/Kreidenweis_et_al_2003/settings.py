from PySDM.physics import si
from PySDM.initialisation import spectral_sampling, spectra
from PySDM.physics import formulae as phys
from PySDM.physics import constants as const
from chempy import Substance
import numpy as np
from collections import OrderedDict  # TODO ???


DRY_RHO = 1800 * si.kg / (si.m ** 3)
DRY_FORMULA = "NH4HSO4"
DRY_SUBSTANCE = Substance.from_formula(DRY_FORMULA)


def default_init(dry_v):
    return np.zeros_like(dry_v)


def compound_init(dry_v):
    return dry_v_to_amount(dry_v)


def dry_v_to_amount(v):
    return (v * DRY_RHO / (DRY_SUBSTANCE.mass * si.gram / si.mole))


COMPOUNDS = OrderedDict({
    "SO2": default_init,
    "O3": default_init,
    "H2O2": default_init,
    "CO2": default_init,
    "HNO3": default_init,
    "NH3": compound_init,
    "HSO4m": compound_init,
    # "Hp": lambda d, w: default_init(d, w) + 1e-7 * M * w,
    "Hp": compound_init,
})


def get_starting_amounts(dry_v):
    return {k: v(dry_v) for k, v in COMPOUNDS.items()}


class Settings:
    def __init__(self, dt, n_sd):
        self.system_type = 'closed'
        self.t_max = (2400 + 196) * si.s
        self.w = .5 * si.m/si.s
        self.g = 10 * si.m / si.s**2

        self.dt = dt
        self.n_sd = n_sd

        self.p0 = 950 * si.mbar
        self.T0 = 285.2 * si.K
        self.pv0 = .95 * phys.pvs(self.T0)

        self.kappa = .61  # TODO
        rho = 1  # TODO
        self.mass_of_dry_air = 44  # TODO

        self.r_dry, self.n_in_dv = spectral_sampling.ConstantMultiplicity(
            spectrum=spectra.Lognormal(
                norm_factor=566 / si.cm**3 / rho * self.mass_of_dry_air,
                m_mode=.08 * si.um / 2,
                s_geom=2
            )
        ).sample(n_sd)

        self.ENVIRONMENT_MOLE_FRACTIONS = {
            "SO2": 0.2 * const.ppb,
            "O3": 50 * const.ppb,
            "H2O2": 0.5 * const.ppb,
            "CO2": 360 * const.ppm,
            "HNO3": 0.1 * const.ppb,
            "NH3": 0.1 * const.ppb,
        }

        self.starting_amounts = get_starting_amounts(phys.volume(self.r_dry))

    @property
    def nt(self):
        nt = self.t_max / self.dt
        assert nt == int(nt)
        return int(nt)
