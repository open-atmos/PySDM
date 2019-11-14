import pint
from PySDM.simulation.fake_unit_registry import FakeUnitRegistry
import numpy as np
import numba
import mendeleev as pt
from scipy import constants as sci

# si = pint.UnitRegistry()
si = FakeUnitRegistry(pint.UnitRegistry())
mgn = lambda x: x
# mgn = lambda x: x.to_base_units().magnitude
awgh = lambda x: x.atomic_weight * si.gram / si.mole

th_dry = lambda th_std, qv: th_std * np.power(1 + qv / eps, Rd / c_pd)

Md = 0.78 * awgh(pt.N) * 2 + 0.21 * awgh(pt.O) * 2 + 0.01 * awgh(pt.Ar)
Mv = awgh(pt.O) + awgh(pt.H) * 2

R_str = sci.R * si.joule / si.kelvin / si.mole
eps = Mv / Md
Rd = R_str / Md
Rv = R_str / Mv
D0 = 2.26e-5 * si.metre ** 2 / si.second
K0 = 2.4e-2 * si.joules / si.metres / si.seconds / si.kelvins

mix = lambda q, dry, wet: wet / (1 / q + 1) + dry / (1 + q)
c_p = lambda q: mix(q, c_pd, c_pv)
R = lambda q: mix(q, Rd, Rv)

p1000 = 1000 * si.hectopascals
c_pd = 1005 * si.joule / si.kilogram / si.kelvin
c_pv = 1850 * si.joule / si.kilogram / si.kelvin
T0 = sci.zero_Celsius * si.kelvin
g = sci.g * si.metre / si.second ** 2

c_pw = 4218 * si.joule / si.kilogram / si.kelvin

ARM_C1 = 6.1094 * si.hectopascal
ARM_C2 = 17.625 * si.dimensionless
ARM_C3 = 243.04 * si.kelvin

rho_w = 1 * si.kilograms / si.litres
sgm = 0.072 * si.joule / si.metre ** 2  # TODO: temperature dependence

p_tri = 611.73 * si.pascal
T_tri = 273.16 * si.kelvin
l_tri = 2.5e6 * si.joule / si.kilogram


@numba.njit()
def dr_dt_MM(r, T, p, S, kp, rd):
    nom = (S - A(T) / r + B(kp, rd) / r ** 3)
    den = Fd(T, D(r, T)) + Fk(T, K(r, T, p), lv(T))
    result = 1 / r * nom / den
    return result

@numba.njit()
def lv(T):
    # latent heat of evaporation
    return l_tri + (c_pv - c_pw) * (T - T_tri)

@numba.njit()
def lambdaD(T):
    return D0 / np.sqrt(2 * Rv * T)

@numba.njit()
def lambdaK(T, p):
    return (4 / 5) * K0 * T / p / np.sqrt(2 * Rd * T)

@numba.njit()
def beta(Kn):
    return (1 + Kn) / (1 + 1.71 * Kn + 1.33 * Kn * Kn)

@numba.njit()
def D(r, T):
    Kn = lambdaD(T) / r  # TODO: optional
    return D0 * beta(Kn)

@numba.njit()
def K(r, T, p):
    Kn = lambdaK(T, p) / r
    return K0 * beta(Kn)

# Maxwel-Mason coefficients:
@numba.njit()
def Fd(T, D):
    return rho_w * Rv * T / D / pvs(T)

@numba.njit()
def Fk(T, K, lv):
    return rho_w * lv / K / T * (lv / Rv / T - 1)

# Koehler curve (expressed in partial pressure):
@numba.njit()
def A(T):
    return 2 * sgm / Rv / T / rho_w

@numba.njit()
def B(kp, rd):
    return kp * rd ** 3

@numba.njit()
def r_cr(kp, rd, T):
    # critical radius
    return np.sqrt(3 * kp * rd ** 3 / A(T))

@numba.njit()
def pvs(T):
    # August-Roche-Magnus formula
    return ARM_C1 * np.exp((ARM_C2 * (T - T0)) / (T - T0 + ARM_C3))