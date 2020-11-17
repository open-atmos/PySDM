"""
Crated at 2019
"""

import molmass
import pint
from scipy import constants as sci
from PySDM.physics._fake_unit_registry import FakeUnitRegistry
from PySDM.physics._flag import DIMENSIONAL_ANALYSIS

si = pint.UnitRegistry()
if not DIMENSIONAL_ANALYSIS:
    si = FakeUnitRegistry(si)


def _weight(x):
    return molmass.Formula(x).mass * si.gram / si.mole


def convert_to(value, unit):
    value /= unit


pi = sci.pi

Md = 0.78 * _weight('N2') + 0.21 * _weight('O2') + 0.01 * _weight('Ar')
Mv = _weight('H2O')

R_str = sci.R * si.joule / si.kelvin / si.mole
eps = Mv / Md
Rd = R_str / Md
Rv = R_str / Mv
D0 = 2.26e-5 * si.metre ** 2 / si.second
K0 = 2.4e-2 * si.joules / si.metres / si.seconds / si.kelvins

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

# standard pressure and temperature (ICAO)
T_STP = T0 + 15 * si.kelvin
p_STP = 101325 * si.pascal
rho_STP = p_STP / Rd / T_STP
