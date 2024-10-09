"""
transition-regime correction as in 'Physics and Chemistry of Clouds' by Lamb and Verlinde (2011), Chapter 8.2  
or 13.1 in Pruppbacher and Klett (2010)
with free pathway of air/vapour (lambdaD) after Pruppacher and Klett (2010)
"""

import numpy as np


class Standard:
    def __init__(self, _):
        pass
    
    @staticmethod
    def lambdaD(_, T, p):  # pylint: disable=unused-argument
        return  const.lmbd_w_0 * T / const.T_STP  * const.p_STP / p

    @staticmethod
    def lambdaK(_, T, p):  # pylint: disable=unused-argument
        return const.lmbd_w_0 * T / const.T_STP  * const.p_STP / p

    @staticmethod
    def D(_, D, r, lmbd, T):  # pylint: disable=unused-argument
        return D / ( r / (r + lmbd) + 4.0 * D /  const.MAC_ice / np.sqrt(8. * const.Rv * T / const.PI) / r  )

    @staticmethod
    def K(_, K, r, lmbd, T, rho):  # pylint: disable=unused-argument
        return K / ( r / (r + lmbd) + 4.0 * K / const.HAC_ice / np.sqrt(8. * const.Rd * T / const.PI)    / const.c_pd / rho /  r )
