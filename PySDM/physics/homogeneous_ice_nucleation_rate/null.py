"""
do-nothing null formulation (needed as other formulations require parameters
 to be set before instantiation of Formulae)
"""

import numpy as np


class Null:  # pylint: disable=too-few-public-methods,unused-argument
    def __init__(self, _):
        pass

    @staticmethod
    def j_hom(const, T, d_a_w_ice):  # pylint: disable=unused-argument
        return np.nan
