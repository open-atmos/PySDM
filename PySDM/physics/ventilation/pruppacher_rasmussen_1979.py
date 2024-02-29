"""
ventilation coefficient as a function of dimensionless Reynolds (Re) and Schmidt (Sc)
numbers for liquid drops following
[Pruppacher & Rasmussen (1979)](https://doi.org/10.1175/1520-0469(1979)036<1255:AWTIOT>2.0.CO;2)
NB: this parameterization is only experimentally validated for Re < 2600
but is hypothesized to be valid for spheres with Re < 8 × 10⁴
based on theory (Pruppacher & Rasmussen, 1979).
the parameterization also does not account for effects of air turbulence.
"""

import numpy as np


class PruppacherAndRasmussen1979:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def ventilation_coefficient(const, sqrt_re_times_cbrt_sc):
        return np.where(
            sqrt_re_times_cbrt_sc < const.PRUPPACHER_RASMUSSEN_1979_XTHRES,
            (
                const.PRUPPACHER_RASMUSSEN_1979_CONSTSMALL
                + const.PRUPPACHER_RASMUSSEN_1979_COEFFSMALL
                * np.power(
                    sqrt_re_times_cbrt_sc, const.PRUPPACHER_RASMUSSEN_1979_POWSMALL
                )
            ),
            (
                const.PRUPPACHER_RASMUSSEN_1979_CONSTBIG
                + const.PRUPPACHER_RASMUSSEN_1979_COEFFBIG * sqrt_re_times_cbrt_sc
            ),
        )
