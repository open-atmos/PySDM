"""
[Bolton 1980](https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2)
"""

import numpy as np


class Bolton1980:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_Celsius(const, T):
        """valid for 0 <= T <= 100 C, eq (9)"""
        return (
            np.exp((T**-2) * const.B80W_G0)
            + ((T**-1) * const.B80W_G1)
            + (T * const.B80W_G2)
            + ((T**1) * const.B80W_G3)
            + ((T**2) * const.B80W_G4)
            + ((T**3) * const.B80W_G5)
            + ((T**4) * const.B80W_G6)
            + (np.log(T) * const.B80W_G7)
        )
