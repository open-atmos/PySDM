"""
as in [Pruppacher and Klett 2005, chapter 13](https://doi.org/10.1007/978-0-306-48100-0_13),
eq. (13-14) - with reference to [Okuyama and Zung 1967](https://doi.org/10.1063/1.1840906)
"""

import numpy as np


class PruppacherKlett:
    """D() stands for modified diffusivity, lamdaD stands for TODO"""

    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(const, D, T):
        return D / np.sqrt(2 * const.Rv * T)

    @staticmethod
    def D(const, D, r, lmbd):
        return D / (
            (r / (r + const.dv_pk05)) + 2 * np.sqrt(const.PI) * lmbd / r / const.MAC
        )
