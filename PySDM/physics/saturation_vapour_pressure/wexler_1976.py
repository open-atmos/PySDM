"""
[Wexler 1976](https://nvlpubs.nist.gov/nistpubs/jres/80A/jresv80An5-6p775_A1b.pdf)
"""

import numpy as np


class Wexler1976:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_Celsius(const, T):
        return (
            np.exp(
                const.W76W_G0 / (T + const.T0) ** 2
                + const.W76W_G1 / (T + const.T0)
                + const.W76W_G2
                + const.W76W_G3 * (T + const.T0)
                + const.W76W_G4 * (T + const.T0) ** 2
                + const.W76W_G5 * (T + const.T0) ** 3
                + const.W76W_G6 * (T + const.T0) ** 4
                + const.W76W_G7 * np.log((T + const.T0) / const.one_kelvin)
            )
            * const.W76W_G8
        )

    @staticmethod
    def ice_Celsius(const, T):
        """NaN with unit of pressure and correct dimension"""
        return np.nan * T / const.B80W_G2 * const.B80W_G0
