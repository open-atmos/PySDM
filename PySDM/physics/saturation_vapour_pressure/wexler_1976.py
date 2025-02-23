"""
[Wexler 1976](https://doi.org/10.6028/jres.080A.071)
"""

import numpy as np


class Wexler1976:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_water(const, T):
        return (
            np.exp(
                const.W76W_G0 / T**2
                + const.W76W_G1 / T
                + const.W76W_G2
                + const.W76W_G3 * T
                + const.W76W_G4 * T**2
                + const.W76W_G5 * T**3
                + const.W76W_G6 * T**4
                + const.W76W_G7 * np.log(T / const.one_kelvin)
            )
            * const.W76W_G8
        )

    @staticmethod
    def pvs_ice(const, T):
        """NaN with unit of pressure and correct dimension"""
        return np.nan * (T - const.T0) / const.B80W_G2 * const.B80W_G0
