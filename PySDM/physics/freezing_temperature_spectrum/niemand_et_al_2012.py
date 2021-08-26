from PySDM.physics import constants as const, si
import numpy as np

a = -0.517
b = 8.934
A_insol = const.pi * (1*si.um)**2

class Niemand_et_al_2012:
    def __str__(self):
        return 'Niemand et al. 2012'

    @staticmethod
    def pdf(T):
        ns_T = np.exp(a * (T - const.T0) + b)
        return -A_insol * a * ns_T * np.exp(-A_insol * ns_T)

    @staticmethod
    def cdf(T):
        ns_T = np.exp(a * (T - const.T0) + b)
        return 1 - np.exp(-A_insol * ns_T) - np.exp(-A_insol*np.exp(-a * const.T0 + b))
