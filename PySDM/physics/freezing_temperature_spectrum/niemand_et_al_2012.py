import numpy as np
from PySDM.physics import constants as const

a = np.nan
b = np.nan


class Niemand_et_al_2012:
    def __str__(self):
        return 'Niemand et al. 2012'

    def __init__(self):
        assert np.isfinite(a)
        assert np.isfinite(b)

    @staticmethod
    def pdf(T, A_insol):
        ns_T = np.exp(a * (T - const.T0) + b)
        return -A_insol * a * ns_T * np.exp(-A_insol * ns_T)

    @staticmethod
    def cdf(T, A_insol):
        ns_T = np.exp(a * (T - const.T0) + b)
        return 1 - np.exp(-A_insol * ns_T) - np.exp(-A_insol*np.exp(-a * const.T0 + b))

    @staticmethod
    def invcdf(cdf, A_insol):
        tmp = np.log((np.log(1 - cdf) + np.exp(-A_insol*np.exp(-a * const.T0 + b))) / -A_insol)
        return const.T0 + (tmp - b) / a
