"""
[Bolton 1980](https://doi.org/10.1175/1520-0493(1980)108%3C1046:TCOEPT%3E2.0.CO;2)
"""

import numpy as np


class Bolton1980:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_water(const, T):
        """valid for 243.15(-30) <= T <= 308.15(35) K(C), eq. (10)"""
        return const.B80W_G0 * np.exp(
            (const.B80W_G1 * (T - const.T0)) / ((T - const.T0) + const.B80W_G2)
        )

    @staticmethod
    def pvs_ice(const, T):
        """NaN with unit of pressure and correct dimension"""
        return np.nan * (T - const.T0) / const.B80W_G2 * const.B80W_G0
