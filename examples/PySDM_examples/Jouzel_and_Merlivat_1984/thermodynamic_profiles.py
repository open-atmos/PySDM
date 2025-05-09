"""
From [Jouzel & Merlivat 1984](https://doi.org/10.1029/JD089iD07p11749).
"""

import numpy as np
from scipy.interpolate import make_interp_spline

from PySDM.physics.constants import T0
from PySDM.physics import si
from PySDM import Formulae


pressure = make_interp_spline(
    x=np.asarray([-10, -20, -30, -40, -50])[::-1] + T0,
    y=np.asarray([925, 780, 690, 630, 600])[::-1] * si.mbar,
)
""" Table 1, first two columns: temperature and pressure"""
pressure.extrapolate = False


def ice_saturation_curve_4(const, T):
    """eq. (15)"""
    return (
        const.JOUZEL_MERLIVAT_CURVE_4_INTERCEPT_COEFF
        + const.JOUZEL_MERLIVAT_CURVE_4_SLOPE_COEFF * (T - const.T0)
    )


def vapour_mixing_ratio(formulae, T):
    """mixing ratio with saturation wrt ice calculated with curve 4 equation"""
    const = formulae.constants
    svp = formulae.saturation_vapour_pressure
    p_v = ice_saturation_curve_4(const, T) * svp.pvs_ice(T)
    rho_v = p_v / const.Rv / T
    p_d = pressure(T) - p_v
    rho_d = p_d / const.Rd / T
    return rho_v / rho_d
