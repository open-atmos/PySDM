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
