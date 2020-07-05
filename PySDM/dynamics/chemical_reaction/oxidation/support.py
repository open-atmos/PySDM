"""
Created at 18.05.2020

@author: Grzegorz Łazarski
"""

import numpy as np

from PySDM.physics import formulae as fml
from PySDM.physics.constants import R_str, pi

from .constants import K_H2O, ROOM_TEMP, DRY_RHO, DRY_SUBSTANCE, gpm_u, M


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


def hydrogen_conc_factory(*, NH3, HNO3, CO2, SO2, HSO4m, K_HNO3, K_SO2,
                          K_NH3, K_CO2, K_HSO3, K_HCO3, K_HSO4, **kwargs):
    N_III, N_V, C, S_IV, S_VI = NH3, HNO3, CO2, SO2, HSO4m

    def concentration(H):
        ammonia = (N_III * H * K_NH3) / (K_H2O + K_NH3 * H)
        nitric = N_V * K_HNO3 / (H + K_HNO3)
        sulfous = S_IV * K_SO2 * (H + 2*K_HSO3) / \
            (H * H + H * K_SO2 + K_SO2 * K_HSO3)
        water = K_H2O / H
        sulfuric = S_VI * (H + 2 * K_HSO4) / (H + K_HSO4)
        carbonic = C * K_CO2 * (H + 2 * K_HCO3) / \
            (H * H + H * K_CO2 + K_CO2 * K_HCO3)
        result = H + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
        return result

    return concentration


def oxidation_factory(*, k0, k1, k2, k3, K_SO2, K_HSO3, magic_const=13 / M, **kwargs):
    # NB: magic_const in the paper is k4.
    # The value is fixed at 13 M^-1 (from dr Jaruga's Thesis)

    # NB: This might not be entirely correct
    # https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD092iD04p04171
    # https://www.atmos-chem-phys.net/16/1693/2016/acp-16-1693-2016.pdf

    # NB: There is also slight error due to "borrowing" compounds when
    # the concentration is close to 0. That way, if the rate is big enough,
    # it will consume more compound than there is.

    def oxidation(x):
        x[x < 0] = 0
        H, O3, SO2aq, H2O2, HSO4m = x

        ozone = ((k0) + (k1 * K_SO2 / H) + (k2 * K_SO2
                                            * K_HSO3 / (H ** 2))) * O3 * SO2aq
        peroxide = k3 * K_SO2 / (1 + magic_const * H) * H2O2 * SO2aq

        return (0, -ozone, -(ozone + peroxide), -peroxide, ozone + peroxide)
    return oxidation


def henry_factory(Da, a_Ma, Ma, Henry, hconcdep):
    def henry(A, cinf, *, T, V_w, H, dt):
        Heff = Henry.at(T) * hconcdep(H)
        r_w = fml.radius(V_w)
        v_avg = np.sqrt(8 * R_str * T / (pi * Ma))
        scale = (4 * r_w / (3 * v_avg * a_Ma) + r_w ** 2 / (3 * Da))
        new_A = (A + dt * cinf / scale) / (1 + dt / (scale * Heff * R_str * T))
        return new_A
    return henry


def calc_ionic_strength(*, Hp, NH3, HNO3, CO2, SO2, HSO4m, K_HNO3, K_SO2,
                        K_NH3, K_CO2, K_HSO3, K_HCO3, K_HSO4, **kwargs):
    N_III, N_V, C_IV, S_IV, S_VI = NH3, HNO3, CO2, SO2, HSO4m

    # Directly adapted
    # https://github.com/igfuw/libcloudphxx/blob/0b4e2455fba4f95c7387623fc21481a85e7b151f/src/impl/particles_impl_chem_strength.ipp#L50
    # https://en.wikipedia.org/wiki/Ionic_strength

    # H+ and OH-
    water = Hp + K_H2O / Hp

    # HSO4- and SO4 2-
    czS_VI = Hp * S_VI / (Hp + K_HSO4) + 4 * K_HSO4 * S_VI / (Hp + K_HSO4)

    # HCO3- and CO3 2-
    cz_CO2 = K_CO2 * Hp * C_IV / (Hp * Hp + K_CO2 * Hp + K_CO2 * K_HCO3) + \
        4 * K_CO2 * K_HCO3 * C_IV / (Hp * Hp + K_CO2 * Hp + K_CO2 * K_HCO3)

    # HSO3- and HSO4 2-
    cz_SO2 = K_SO2 * Hp * S_IV / (Hp * Hp + K_SO2 * Hp + K_SO2 * K_HSO3) + \
        4 * K_SO2 * K_HSO3 * S_IV / (Hp * Hp + K_SO2 * Hp + K_SO2 * K_HSO3)

    # NO3-
    cz_HNO3 = K_HNO3 * N_V / (Hp + K_HNO3)

    # NH4+
    cz_NH3 = K_NH3 * Hp * N_III / (K_H2O + K_NH3 * Hp)

    return 0.5 * (water + czS_VI + cz_CO2 + cz_SO2 + cz_HNO3 + cz_NH3)


def dry_v_to_amount(v):
    return (v * DRY_RHO / (DRY_SUBSTANCE.mass * gpm_u))


def amount_to_dry_v(amnt):
    return ((amnt * DRY_SUBSTANCE.mass * gpm_u) / DRY_RHO)
