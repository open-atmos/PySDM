"""
Created at 11.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.attributes.derived_attribute import DerivedAttribute
from .radius import Radius
from PySDM.physics import constants as const


class TerminalVelocity(DerivedAttribute):

    def __init__(self, particles_builder):
        self.radius = particles_builder.get_instance(Radius)
        dependencies = [self.radius]
        super().__init__(particles_builder, name='terminal velocity', dependencies=dependencies)
        self.k1 = 1.19e6 / const.si.centimetre / const.si.second
        self.k2 = 8e3 / const.si.second
        self.k3 = 2.01e3 * const.si.centimetre ** (1 / 2) / const.si.second
        self.r1 = 40 * const.si.um
        self.r2 = 600 * const.si.um

    def recalculate(self):
        numba_term_vel(self.data, self.radius.get(), self.k1, self.k2, self.k3, self.r1, self.r2)


# TODO: move to backend
import numba
@numba.njit()
def numba_term_vel(values, radius, k1, k2, k3, v1, v2):
    for i in range(len(values)):
        if radius[i] < v1:
            values[i] = k1 * radius[i] ** 2
        elif radius[i] < v2:
            values[i] = k2 * radius[i]
        else:
            values[i] = k3 * radius[i] ** (1 / 2)
