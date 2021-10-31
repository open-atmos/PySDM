from numpy import exp
from PySDM.physics import constants as const


class AugustRocheMagnus:
    @staticmethod
    def pvs_Celsius(T):
        return const.ARM_C1 * exp((const.ARM_C2 * T) / (T + const.ARM_C3))
