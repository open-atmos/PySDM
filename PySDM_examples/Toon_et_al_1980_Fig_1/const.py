import pint
from scipy import constants as sci
from PySDM.physics._fake_unit_registry import FakeUnitRegistry
from PySDM.physics._flag import DIMENSIONAL_ANALYSIS

si = pint.UnitRegistry()
if not DIMENSIONAL_ANALYSIS:
    si = FakeUnitRegistry(si)

'''def g(z):
    return g0 * (Rg / (Rg + z))**2

g0 = 1.17 * si.centimeter / si.second **2
Rg = 2.8 * 10**8 * si.centimeter
R_str = sci.R * si.joule / si.kelvin / si.mole

u = 16 * si.gram / si.mol
Rd = R_str / u'''
# ================== NOWE =====================================


class PlanetData():
    def __init__(self, g0, Rg, Rd):
        self.g0 = g0
        self.Rg = Rg
        self.Rd = Rd

    def g(self, z):
        acc = self.g0 * (self.Rg / (self.Rg + z)) ** 2
        return acc


class Planets():
    def __init__(self):
        self.__add_earth()
        self.__add_titan()

    def __add_earth(self):
        "Dane dla Ziemi"
        self.Earth = PlanetData(1, 2, 3)

    def __add_titan(self):
        g0 = 1.17 * si.centimeter / si.second ** 2
        Rg = 2.8 * 10 ** 8 * si.centimeter
        R_str = sci.R * si.joule / si.kelvin / si.mole
        u = 16 * si.gram / si.mol
        Rd = R_str / u
        self.Titan = PlanetData(g0, Rg, Rd)


#const2 = Planets()
#print(const2.Titan.Rd)
# print(const.Earth.R)

