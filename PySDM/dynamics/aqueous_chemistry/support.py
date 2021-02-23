from PySDM.physics.constants import R_str, ROOM_TEMP
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


def hydrogen_conc_factory(*, NH3, HNO3, CO2, SO2, HSO4m, K_HNO3, K_SO2,
                          K_NH3, K_CO2, K_HSO3, K_HCO3, K_HSO4, **kwargs):
    N_III, N_V, C, S_IV, S_VI = NH3/v, HNO3/v, CO2/v, SO2/v, HSO4m/v

    def concentration(H):
        H /= v

        ammonia = (N_III * H * K_NH3) / (K_H2O + K_NH3 * H)
        nitric = N_V * K_HNO3 / (H + K_HNO3)
        sulfous = S_IV * K_SO2 * (H + 2*K_HSO3) / \
            (H * H + H * K_SO2 + K_SO2 * K_HSO3)
        water = K_H2O / H
        sulfuric = S_VI * (H + 2 * K_HSO4) / (H + K_HSO4)
        carbonic = C * K_CO2 * (H + 2 * K_HCO3) / \
            (H * H + H * K_CO2 + K_CO2 * K_HCO3)
        zero = H + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
        return zero

    return concentration