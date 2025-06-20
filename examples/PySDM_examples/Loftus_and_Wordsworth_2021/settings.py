# Planetary Properties, Loftus and Wordsworth 2021 Table 1

from pystrict import strict

from PySDM import Formulae
from PySDM.dynamics import condensation
from PySDM.physics.constants import si
from PySDM_examples.Loftus_and_Wordsworth_2021.planet import Planet


@strict
class Settings:
    def __init__(
        self,
        r_wet: float,
        mass_of_dry_air: float,
        planet: Planet,
        initial_water_vapour_mixing_ratio: float,
        pcloud: float,
        Zcloud: float,
        Tcloud: float,
        formulae: Formulae = None,
    ):
        self.formulae = formulae or Formulae(
            saturation_vapour_pressure="AugustRocheMagnus",
            diffusion_coordinate="WaterMassLogarithm",
        )

        self.initial_water_vapour_mixing_ratio = initial_water_vapour_mixing_ratio
        self.p0 = planet.p_STP
        self.RH0 = planet.RH_zref
        self.kappa = 0.2
        self.T0 = planet.T_STP
        self.z_half = 150 * si.metres
        self.dt = 1 * si.second
        self.pcloud = pcloud
        self.Zcloud = Zcloud
        self.Tcloud = Tcloud

        self.r_wet = r_wet
        self.mass_of_dry_air = mass_of_dry_air
        self.n_output = 500

        self.rtol_x = 0.5 * (condensation.DEFAULTS.rtol_x)
        self.rtol_thd = condensation.DEFAULTS.rtol_thd
        self.dt_cond_range = condensation.DEFAULTS.cond_range
