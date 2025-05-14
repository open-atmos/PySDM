"""
From [Jouzel & Merlivat 1984](https://doi.org/10.1029/JD089iD07p11749).
"""

import numpy as np
from scipy.interpolate import make_interp_spline

from PySDM.physics.constants_defaults import T0
from PySDM.physics import si


pressure = make_interp_spline(
    x=np.asarray([-10, -20, -30, -40, -50])[::-1] + T0,
    y=np.asarray([785, 670, 620, 570, 500])[::-1] * si.mbar,
)
""" Table 1, first two columns: temperature and pressure"""
pressure.extrapolate = False

A_coefficient = make_interp_spline(
    x=np.asarray([-10, -20, -30, -40, -50])[::-1] + T0,
    y=np.asarray([1.5, 1.25, 1.11, 1.05, 1.02])[::-1],
)
""" Table 1, first two columns: temperature and the Thermodynamic Coefficient A"""


def ice_saturation_curve_4(const, T):
    """eq. (15)"""
    return (
        const.JOUZEL_MERLIVAT_CURVE_4_INTERCEPT_COEFF
        - const.JOUZEL_MERLIVAT_CURVE_4_SLOPE_COEFF * (T - const.T0)
    )


def vapour_mixing_ratio(const, T, svp):
    """mixing ratio with saturation wrt ice calculated with curve 4 equation"""
    p_v = ice_saturation_curve_4(const, T) * svp.pvs_ice(T)
    rho_v = p_v / const.Rv / T
    rho_d = const.rho_STP
    return rho_v / rho_d
