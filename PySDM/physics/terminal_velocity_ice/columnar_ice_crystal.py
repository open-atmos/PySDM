"""
[Spichtinger & Gierens 2009](https://doi.org/10.5194/acp-9-685-2009)
Eq. (18) and (19) Assumed shape is columnar based on empirical parameterizations of
[Heymsfield & Iaquinta (2000)](https://doi.org/10.1175/1520-0469(2000)057%3C0916:CCTV%3E2.0.CO;2)
[Barthazy & Schefold (2006)](https://doi.org/10.1016/j.atmosres.2005.12.009)
"""

import numpy as np


class ColumnarIceCrystal:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def v_base_term(const, mass):
        return np.where(
            mass < const.SPICHTINGER_GIERENS_TERM_VEL_LIMIT_0,
            const.SPICHTINGER_TERM_GAMMA_COEFF0
            * mass**const.SPICHTINGER_TERM_DELTA_COEFF0,
            np.where(
                mass < const.SPICHTINGER_GIERENS_TERM_VEL_LIMIT_1,
                const.SPICHTINGER_TERM_GAMMA_COEFF1
                * mass**const.SPICHTINGER_TERM_DELTA_COEFF1,
                np.where(
                    mass < const.SPICHTINGER_GIERENS_TERM_VEL_LIMIT_2,
                    const.SPICHTINGER_TERM_GAMMA_COEFF2
                    * mass**const.SPICHTINGER_TERM_DELTA_COEFF2,
                    const.SPICHTINGER_TERM_GAMMA_COEFF3
                    * mass**const.SPICHTINGER_TERM_DELTA_COEFF3,
                ),
            ),
        )

    @staticmethod
    def atmospheric_correction_factor(const, temperature, pressure):
        return (
            pressure / const.SPICHTINGER_CORRECTION_P0
        ) ** const.SPICHTINGER_CORRECTION_P_EXPO * (
            const.SPICHTINGER_CORRECTION_T0 / temperature
        ) ** const.SPICHTINGER_CORRECTION_T_EXPO
