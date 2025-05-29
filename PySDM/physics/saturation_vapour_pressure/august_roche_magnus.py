"""
August-Roche-Magnus formula (see, e.g.,
[Wikipedia](https://en.wikipedia.org/wiki/Clausius–Clapeyron_relation#August–Roche–Magnus_formula)
and references therein)
"""

import numpy as np


class AugustRocheMagnus:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_water(const, T):
        return const.ARM_C1 * np.exp(
            (const.ARM_C2 * (T - const.T0)) / ((T - const.T0) + const.ARM_C3)
        )

    @staticmethod
    def pvs_ice(const, T):
        """NaN with unit of pressure and correct dimension"""
        return np.nan * (T - const.T0) / const.ARM_C3 * const.ARM_C1
