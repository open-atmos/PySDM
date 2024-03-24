"""
[Bolton 1980](https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2)
"""

import numpy as np


class Bolton1980:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_Celsius(const, T):
        """valid for -30 <= T <= 35 C, eq (10)"""
        return const.B80W_G0 * np.exp((const.B80W_G1 * T) / (T + const.B80W_G2))

    @staticmethod
    def ice_Celsius(const, T):
        """NaN with unit of pressure and correct dimension"""
        return np.nan * T / const.B80W_G2 * const.B80W_G0
