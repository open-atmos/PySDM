"""
Collection of constants with dimensional analysis handled with
[Pint](https://pypi.org/project/Pint/)'s package `UnitRegistry` for test
purposes and mocked with `PySDM.physics.impl.fake_unit_registry.FakeUnitRegistry` by default.
"""

import os
import time

import numpy as np
import pint
from scipy import constants as sci

from .impl.fake_unit_registry import FakeUnitRegistry
from .impl.flag import DIMENSIONAL_ANALYSIS

si = pint.UnitRegistry()
if not DIMENSIONAL_ANALYSIS:
    si = FakeUnitRegistry(si)


def convert_to(value, unit):
    value /= unit


def in_unit(value, unit):
    return value / unit


sqrt_two = np.sqrt(2)
sqrt_pi = np.sqrt(sci.pi)
PI = sci.pi
PI_4_3 = PI * 4 / 3
LN_2 = np.log(2)
ZERO_MASS = 0 * si.kg
ZERO_VOLUME = 0 * si.m**3
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
TWELVE = 12
ONE_THIRD = 1 / 3
ONE_HALF = 1 / 2
TWO_THIRDS = 2 / 3
ONE_AND_A_HALF = 3 / 2
TWO_AND_A_HALF = 5 / 2
NaN = np.nan

default_random_seed = (
    44
    if "CI" in os.environ  # https://en.wikipedia.org/wiki/44_(number)
    else time.time_ns()
)

PPT = 1e-12
PPB = 1e-9
PPM = 1e-6

PER_CENT = 1e-2
PER_MILLE = 1e-3
PER_MEG = PPM

T0 = sci.zero_Celsius * si.kelvin

# there are so few water ions instead of K we have K [H2O] (see Seinfeld & Pandis p 345)
M = si.mole / si.litre
K_H2O = 1e-14 * M * M

CM = 1 * si.cm
UM = 1 * si.um
