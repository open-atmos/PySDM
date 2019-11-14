import pint
import numpy as np
import mendeleev as pt
from scipy import constants as sci

si = pint.UnitRegistry()
mgn = lambda x: x.to_base_units().magnitude
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
