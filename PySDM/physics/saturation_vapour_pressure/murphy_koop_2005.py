"""
[Murphy and Koop 2005](https://doi.org/10.1256/qj.04.94)
"""

import numpy as np


class MurphyKoop2005:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_Celsius(const, T):
        """valid for 123 < T < 332 K, eq (10)"""
        return const.MK05_LIQ_C1 * np.exp(
            const.MK05_LIQ_C2
            - const.MK05_LIQ_C3 / (T + const.T0)
            - const.MK05_LIQ_C4 * np.log((T + const.T0) / const.MK05_LIQ_C5)
            + const.MK05_LIQ_C6 * (T + const.T0)
            + np.tanh(const.MK05_LIQ_C7 * (T + const.T0 - const.MK05_LIQ_C8))
            * (
                const.MK05_LIQ_C9
                - const.MK05_LIQ_C10 / (T + const.T0)
                - const.MK05_LIQ_C11 * np.log((T + const.T0) / const.MK05_LIQ_C12)
                + const.MK05_LIQ_C13 * (T + const.T0)
            )
        )

    @staticmethod
    def ice_Celsius(const, T):
        """valid for T > 110 K, eq (7)"""
        return const.MK05_ICE_C1 * np.exp(
            const.MK05_ICE_C2
            - const.MK05_ICE_C3 / (T + const.T0)
            + const.MK05_ICE_C4 * np.log((T + const.T0) / const.MK05_ICE_C5)
            - const.MK05_ICE_C6 * (T + const.T0)
        )
