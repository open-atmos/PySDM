from PySDM.physics import si
from PySDM.physics.constants import R_str, ROOM_TEMP, H_u, dT_u, M, _weight, Md
import numpy as np


def vant_hoff(K, dH, T, *, T_0=ROOM_TEMP):
    return K * np.exp(-dH / R_str * (1 / T - 1/T_0))


def tdep2enthalpy(tdep):
    return -tdep * R_str


class EqConst:
    def __init__(self, constant_at_T0, dT, *, T_0=ROOM_TEMP, enthalpy=False):
        self.K = constant_at_T0
        self.dH = dT if enthalpy else tdep2enthalpy(dT)
        self.T0 = T_0

    def at(self, T):
        return vant_hoff(self.K, self.dH, T, T_0=self.T0)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.K + other.K
        else:
            return self.K + other

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self.K - other.K
        else:
            return self.K - other

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.K * other.K
        else:
            return self.K * other

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return self.K / other.K
        else:
            return self.K / other

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            return other.K - self.K
        else:
            return other - self.K

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            return other.K / self.K
        else:
            return other / self.K

    def __neg__(self):
        return -self.K

    def __repr__(self):
        return f"EqConst({self.K}@{self.T0}, {self.dH})"


def arrhenius(A, Ea, T=ROOM_TEMP):
    return A * np.exp(-Ea / (R_str * T))


class KinConst:
    def __init__(self, A, dT, *, energy=False):
        self.A = A
        self.Ea = dT if energy else tdep2enthalpy(dT)

    @staticmethod
    def from_k(k, dT, *, T_0=ROOM_TEMP, energy=False):
        if not energy:
            Ea = tdep2enthalpy(dT)
        A = k * np.exp(Ea / (R_str * T_0))
        return KinConst(A, Ea, energy=True)

    def at(self, T):
        return arrhenius(self.A, self.Ea, T)

    def __repr__(self):
        return f"KinConst({self.A}, {self.Ea})"


HENRY_CONST = {
    "HNO3": EqConst(2.1e5 * H_u, 0 * dT_u),
    "H2O2": EqConst(7.45e4 * H_u, 7300 * dT_u),
    "NH3":  EqConst(62 * H_u, 4110 * dT_u),
    "SO2":  EqConst(1.23 * H_u, 3150 * dT_u),
    "CO2":  EqConst(3.4e-2 * H_u, 2440 * dT_u),
    "O3":   EqConst(1.13e-2 * H_u, 2540 * dT_u),
}
EQUILIBRIUM_CONST = {  # Reaction Specific units, K
    # ("HNO3(aq) = H+ + NO3-", 15.4, 0),
    "K_HNO3": EqConst(15.4 * M, 0 * dT_u),
    # ("H2SO3(aq) = H+ + HSO3-", 1.54*10**-2 * KU, 1960),
    "K_SO2":  EqConst(1.3e-2 * M, 1960 * dT_u),
    # ("NH4+ = NH3(aq) + H+", 10**-9.25 * M, 0),
    "K_NH3":  EqConst(1.7e-5 * M, -450 * dT_u),
    # ("H2CO3(aq) = H+ + HCO3-", 4.3*10**-7 * KU, -1000),
    "K_CO2":  EqConst(4.3e-7 * M, -1000 * dT_u),
    # ("HSO3- = H+ + SO3-2", 6.6*10**-8 * KU, 1500),
    "K_HSO3": EqConst(6.6e-8 * M, 1500 * dT_u),
    # ("HCO3- = H+ + CO3-2", 4.68*10**-11 * KU, -1760),
    "K_HCO3": EqConst(4.68e-11 * M, -1760 * dT_u),
    # ("HSO4- = H+ + SO4-2", 1.2*10**-2 * KU, 2720),
    "K_HSO4": EqConst(1.2e-2 * M, 2720 * dT_u),
}

# there are so few water ions instead of K we have K [H2O] (see Seinfeld & Pandis p 345)
K_H2O = 1e-14 * M * M

DIFFUSION_CONST = {
    "HNO3": 65.25e-6 * si.m**2 / si.s,
    "H2O2": 87.00e-6 * si.m**2 / si.s,
    "NH3":  19.78e-6 * si.m**2 / si.s,
    "SO2":  10.89e-6 * si.m**2 / si.s,
    "CO2":  13.81e-6 * si.m**2 / si.s,
    "O3":   14.44e-6 * si.m**2 / si.s,
}
MASS_ACCOMMODATION_COEFFICIENTS = {
    "HNO3": 0.05,
    "H2O2": 0.018,
    "NH3":  0.05,
    "SO2":  0.035,
    "CO2":  0.05,
    "O3":   0.00053
}
AQUEOUS_COMPOUNDS = {
    "S_IV": ("SO2 H2O", "HSO3", "SO3"),
    "O3": ("O3",),
    "H2O2": ("H2O2",),
    "C_IV": ("CO2 H2O", "HCO3", "CO3"),
    "N_V": ("HNO3", "NO3"),
    "N_mIII": ("NH4", "H2O NH3"),
    "S_VI": ("SO4", "HSO4"),
    "H": ("H",)
}
GASEOUS_COMPOUNDS = {
    "N_V": "HNO3",
    "H2O2": "H2O2",
    "N_mIII": "NH3",
    "S_IV": "SO2",
    "C_IV": "CO2",
    "O3": "O3"
}


def aqq_CO2(H):
    return 1 + EQUILIBRIUM_CONST["K_CO2"] * (1 / H + EQUILIBRIUM_CONST["K_HCO3"] / (H ** 2))


def aqq_SO2(H):
    return 1 + EQUILIBRIUM_CONST["K_SO2"] * (1 / H + EQUILIBRIUM_CONST["K_HSO3"] / (H ** 2))


def aqq_NH3(H):
    return 1 + EQUILIBRIUM_CONST["K_NH3"] / K_H2O * H


def aqq_HNO3(H):
    return 1 + EQUILIBRIUM_CONST["K_HNO3"] / H


def aqq(_):
    return 1


MEMBER = {
    "CO2": aqq_CO2,
    "SO2": aqq_SO2,
    "NH3": aqq_NH3,
    "HNO3": aqq_HNO3,
    "O3": aqq,
    "H2O2": aqq
}
k_u = 1 / (si.s * M)
KINETIC_CONST = {
    "k0": KinConst.from_k(2.4e4 * k_u, 0 * dT_u),
    "k1": KinConst.from_k(3.5e5 * k_u, -5530 * dT_u),
    "k2": KinConst.from_k(1.5e9 * k_u, -5280 * dT_u),
    # Different unit due to a different pseudo-order of kinetics
    "k3": KinConst.from_k(7.45e9 * k_u / M, -4430 * dT_u),
}
SPECIFIC_GRAVITY = {
    compound: _weight(compound) / Md for compound in {*GASEOUS_COMPOUNDS.values()}
}

for compounds in AQUEOUS_COMPOUNDS.values():
    for compound in compounds:
        SPECIFIC_GRAVITY[compound] = _weight(compound) / Md
