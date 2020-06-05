import pint
from scipy import constants as sci
from PySDM.physics._fake_unit_registry import FakeUnitRegistry
from PySDM.physics._flag import DIMENSIONAL_ANALYSIS

si = pint.UnitRegistry()
if not DIMENSIONAL_ANALYSIS:
    si = FakeUnitRegistry(si)

def g(z):
    return g0 * (Rg / (Rg + z))**2

g0 = 1.17 * si.centimeter / si.second **2
Rg = 2.8 * 10**8 * si.centimeter
R_str = sci.R * si.joule / si.kelvin / si.mole

u = 16 * si.kilogram / si.mol
Rd = R_str / u