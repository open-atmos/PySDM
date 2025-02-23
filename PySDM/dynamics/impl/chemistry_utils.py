"""
aqueous chemistry helper utils including specific gravity constants with
 values obtained using [chempy](https://pythonhosted.org/chempy/)'s `Substance`
"""

import numpy as np
from chempy import Substance

from PySDM.physics.constants import K_H2O, M, si


class EqConst:  # pylint: disable=too-few-public-methods
    def __init__(self, formulae, constant_at_T0, dT, T_0):
        self.formulae = formulae
        self.K = constant_at_T0
        self.dH = formulae.trivia.tdep2enthalpy(dT)
        self.T0 = T_0

    def at(self, T):
        return self.formulae.trivia.vant_hoff(self.K, self.dH, T, T_0=self.T0)


class KinConst:  # pylint: disable=too-few-public-methods
    def __init__(self, formulae, k, dT, T_0):
        self.formulae = formulae
        self.Ea = formulae.trivia.tdep2enthalpy(dT)
        self.A = k * np.exp(self.Ea / (self.formulae.constants.R_str * T_0))

    def at(self, T):
        return self.formulae.trivia.arrhenius(self.A, self.Ea, T)


class HenryConsts:  # pylint: disable=too-few-public-methods
    def __init__(self, formulae):
        const = formulae.constants
        T0 = const.ROOM_TEMP
        self.HENRY_CONST = {
            "HNO3": EqConst(formulae, 2.1e5 * const.H_u, 8700 * const.dT_u, T_0=T0),
            "H2O2": EqConst(formulae, 7.45e4 * const.H_u, 7300 * const.dT_u, T_0=T0),
            "NH3": EqConst(formulae, 62 * const.H_u, 4110 * const.dT_u, T_0=T0),
            "SO2": EqConst(formulae, 1.23 * const.H_u, 3150 * const.dT_u, T_0=T0),
            "CO2": EqConst(formulae, 3.4e-2 * const.H_u, 2440 * const.dT_u, T_0=T0),
            "O3": EqConst(formulae, 1.13e-2 * const.H_u, 2540 * const.dT_u, T_0=T0),
        }


# Table 4 in Kreidenweis et al. 2003
class EquilibriumConsts:  # pylint: disable=too-few-public-methods
    def __init__(self, formulae):
        const = formulae.constants
        T0 = const.ROOM_TEMP
        self.EQUILIBRIUM_CONST = {  # Reaction Specific units, K
            "K_HNO3": EqConst(formulae, 15.4 * const.M, 8700 * const.dT_u, T_0=T0),
            "K_SO2": EqConst(formulae, 1.3e-2 * const.M, 1960 * const.dT_u, T_0=T0),
            "K_NH3": EqConst(formulae, 1.7e-5 * const.M, -450 * const.dT_u, T_0=T0),
            "K_CO2": EqConst(formulae, 4.3e-7 * const.M, -1000 * const.dT_u, T_0=T0),
            "K_HSO3": EqConst(formulae, 6.6e-8 * const.M, 1500 * const.dT_u, T_0=T0),
            "K_HCO3": EqConst(formulae, 4.68e-11 * const.M, -1760 * const.dT_u, T_0=T0),
            "K_HSO4": EqConst(formulae, 1.2e-2 * const.M, 2720 * const.dT_u, T_0=T0),
        }


DIFFUSION_CONST = {
    "HNO3": 65.25e-6 * si.m**2 / si.s,
    "H2O2": 87.00e-6 * si.m**2 / si.s,
    "NH3": 19.78e-6 * si.m**2 / si.s,
    "SO2": 10.89e-6 * si.m**2 / si.s,
    "CO2": 13.81e-6 * si.m**2 / si.s,
    "O3": 14.44e-6 * si.m**2 / si.s,
}

MASS_ACCOMMODATION_COEFFICIENTS = {
    "HNO3": 0.05,
    "H2O2": 0.018,
    "NH3": 0.05,
    "SO2": 0.035,
    "CO2": 0.05,
    "O3": 0.00053,
}

AQUEOUS_COMPOUNDS = {
    "S_IV": ("SO2 H2O", "HSO3", "SO3"),  # rename: SO2 H2O -> H2SO3(aq) ?
    "O3": ("O3",),
    "H2O2": ("H2O2",),
    "C_IV": ("CO2 H2O", "HCO3", "CO3"),  # ditto
    "N_V": ("HNO3", "NO3"),
    "N_mIII": ("NH4", "H2O NH3"),
    "S_VI": ("SO4", "HSO4"),
}

GASEOUS_COMPOUNDS = {
    "N_V": "HNO3",
    "H2O2": "H2O2",
    "N_mIII": "NH3",
    "S_IV": "SO2",
    "C_IV": "CO2",
    "O3": "O3",
}

DISSOCIATION_FACTORS = {
    "CO2": lambda H, eqc, cell_id: 1
    + eqc["K_CO2"].data[cell_id] * (1 / H + eqc["K_HCO3"].data[cell_id] / (H**2)),
    "SO2": lambda H, eqc, cell_id: 1
    + eqc["K_SO2"].data[cell_id] * (1 / H + eqc["K_HSO3"].data[cell_id] / (H**2)),
    "NH3": lambda H, eqc, cell_id: 1 + eqc["K_NH3"].data[cell_id] / K_H2O * H,
    "HNO3": lambda H, eqc, cell_id: 1 + eqc["K_HNO3"].data[cell_id] / H,
    "O3": lambda _, __, ___: 1,
    "H2O2": lambda _, __, ___: 1,
}


class KineticConsts:  # pylint: disable=too-few-public-methods
    def __init__(self, formulae):
        const = formulae.constants
        T0 = const.ROOM_TEMP
        self.KINETIC_CONST = {
            "k0": KinConst(formulae, k=2.4e4 / si.s / M, dT=0 * const.dT_u, T_0=T0),
            "k1": KinConst(formulae, k=3.5e5 / si.s / M, dT=-5530 * const.dT_u, T_0=T0),
            "k2": KinConst(formulae, k=1.5e9 / si.s / M, dT=-5280 * const.dT_u, T_0=T0),
            # Different unit due to a different pseudo-order of kinetics
            "k3": KinConst(
                formulae, k=7.45e7 / si.s / M / M, dT=-4430 * const.dT_u, T_0=T0
            ),
        }


k4 = 13 / M


class SpecificGravities:  # pylint: disable=too-few-public-methods
    def __init__(self, constants):
        self._values = {
            compound: Substance.from_formula(compound).mass
            * si.gram
            / si.mole
            / constants.Md
            for compound in GASEOUS_COMPOUNDS.values()
        }

        for compounds in AQUEOUS_COMPOUNDS.values():
            for compound in compounds:
                self._values[compound] = (
                    Substance.from_formula(compound).mass
                    * si.gram
                    / si.mole
                    / constants.Md
                )

    def __getitem__(self, item):
        return self._values[item]
