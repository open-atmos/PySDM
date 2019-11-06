import pint
import numpy as np
import mendeleev as pt
from scipy import constants as sci

si = pint.UnitRegistry()
mgn = lambda x: x.to_base_units().magnitude
awgh = lambda x: x.atomic_weight * si.gram / si.mole

# TODO: if we want to incclude such things, then rename file...
th_dry = lambda th_std, qv: th_std * np.power(1 + qv / eps, Rd / c_pd)

Md = 0.78 * awgh(pt.N) * 2 + 0.21 * awgh(pt.O) * 2 + 0.01 * awgh(pt.Ar)
Mv = awgh(pt.O) + awgh(pt.H) * 2

R_str = sci.R * si.joule / si.kelvin / si.mole
eps = Mv / Md
Rd = R_str / Md
Rv = R_str / Mv

mix = lambda q, dry, wet: wet / (1 / q + 1) + dry / (1 + q)
c_p = lambda q: mix(q, c_pd, c_pv)
R = lambda q: mix(q, Rd, Rv)

p1000 = 1000 * si.hectopascals
c_pd = 1005 * si.joule / si.kilogram / si.kelvin
c_pv = 1850 * si.joule / si.kilogram / si.kelvin
T0 = sci.zero_Celsius * si.kelvin
g = sci.g * si.metre / si.second ** 2

ARM_C1 = 6.1094 * si.hectopascal
ARM_C2 = 17.625 * si.dimensionless
ARM_C3 = 243.04 * si.kelvin
