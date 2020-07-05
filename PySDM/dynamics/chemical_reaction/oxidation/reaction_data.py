"""
Created at 18.05.2020

@author: Grzegoprz Łazarski
"""

from collections import defaultdict

from .constants import K_H2O, M, Da_u, H_u, dT_u, k_u, ppb, ppm
from .support import EqConst, KinConst

ENVIRONMENT_AMOUNTS = {
    "SO2": 0.2 * ppb,
    "O3": 50 * ppb,
    "H2O2": 0.5 * ppb,
    "CO2": 360 * ppm,
    "HNO3": 0.1 * ppb,
    "NH3": 0.1 * ppb,
}

EQUILIBRIUM_CONST = {  # Reaction Specific units, K
    # ("HNO3(aq) = H+ + NO3-", 15.4, 0),
    "K_HNO3": EqConst(15.4 * M, 0 * dT_u),
    # ("H2SO3(aq) = H+ + HSO3-", 1.54*10**-2 * KU, 1960),
    "K_SO2":  EqConst((1.3 * 10 ** -2) * M, 1960 * dT_u),
    # ("NH4+ = NH3(aq) + H+", 10**-9.25 * M, 0),
    "K_NH3":  EqConst((1.7 * 10 ** -5) * M, -450 * dT_u),
    # ("H2CO3(aq) = H+ + HCO3-", 4.3*10**-7 * KU, -1000),
    "K_CO2":  EqConst((4.3 * 10 ** -7) * M, -1000 * dT_u),
    # ("HSO3- = H+ + SO3-2", 6.6*10**-8 * KU, 1500),
    "K_HSO3": EqConst((6.6 * 10 ** -8) * M, 1500 * dT_u),
    # ("HCO3- = H+ + CO3-2", 4.68*10**-11 * KU, -1760),
    "K_HCO3": EqConst((4.68 * 10 ** -11) * M, -1760 * dT_u),
    # ("HSO4- = H+ + SO4-2", 1.2*10**-2 * KU, 2720),
    "K_HSO4": EqConst((1.2 * 10 ** -2) * M, 2720 * dT_u)
}

HENRY_CONST = {
    "HNO3": EqConst((2.10 * 10 ** 5) * H_u, 0 * dT_u),
    "H2O2": EqConst((7.45 * 10 ** 4) * H_u, 7300 * dT_u),
    "NH3":  EqConst(62 * H_u, 4110 * dT_u),
    "SO2":  EqConst(1.23 * H_u, 3150 * dT_u),
    "CO2":  EqConst((3.4 * 10 ** -2) * H_u, 2440 * dT_u),
    "O3":   EqConst((1.13 * 10 ** -2) * H_u, 2540 * dT_u),
}

HENRY_PH_DEP = defaultdict(lambda: lambda x: 1, {
    "CO2": lambda H: (1 + EQUILIBRIUM_CONST["K_CO2"]
                      * (1 / H + EQUILIBRIUM_CONST["K_HCO3"] / (H ** 2))),
    "SO2": lambda H: (1 + EQUILIBRIUM_CONST["K_SO2"]
                      * (1 / H + EQUILIBRIUM_CONST["K_HSO3"] / (H ** 2))),
    "NH3": lambda H: (1 + EQUILIBRIUM_CONST["K_NH3"] / K_H2O * H),
    "HNO3": lambda H: (1 + EQUILIBRIUM_CONST["K_HNO3"] / H),
})

DIFFUSION_CONST = {
    "HNO3": ((65.25 * 10 ** -6) * Da_u, 0.05),
    "H2O2": ((87.00 * 10 ** -6) * Da_u, 0.018),
    "NH3":  ((19.78 * 10 ** -6) * Da_u, 0.05),
    "SO2":  ((10.89 * 10 ** -6) * Da_u, 0.035),
    "CO2":  ((13.81 * 10 ** -6) * Da_u, 0.05),
    "O3":   ((14.44 * 10 ** -6) * Da_u, 0.00053),
}

KINETIC_CONST = {
    "k0": KinConst.from_k(2.4 * 10 ** 4 * k_u, 0 * dT_u),
    "k1": KinConst.from_k(3.5 * 10 ** 5 * k_u, -5530 * dT_u),
    "k2": KinConst.from_k(1.5 * 10 ** 9 * k_u, -5280 * dT_u),

    # Different unit due to a different pseudoorder of kinetics
    "k3": KinConst.from_k(7.45 * 10 ** 9 * k_u / M, -4430 * dT_u),
}
