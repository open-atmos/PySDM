import numpy as np
from mpmath.matrices.eigen_symmetric import svd_c

from PySDM.physics.constants_defaults import T0
from scipy.interpolate import make_interp_spline
from PySDM.physics import si


pressure = make_interp_spline(
    x=np.asarray([-10, -20, -30, -40, -50])[::-1] + T0,
    y=np.asarray([925, 780, 690, 630, 600])[::-1] * si.mbar,
)
pressure.extrapolate = False
""" Table 1 """


def ice_saturation_curve_4(T):
    return 0.99 - 0.006 * (T - T0)


def vapour_mixing_ratio(formulae, T):
    const = formulae.constants
    svp = formulae.saturation_vapour_pressure

    p_v = ice_saturation_curve_4(T) * svp.pvs_ice(T)
    rho_v = p_v / const.R_v / T
    p_d = pressure(T) - p_v
    rho_d = p_d / const.R_d / T
    return rho_v / rho_d
