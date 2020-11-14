"""
Created at 05.07.2020
"""

from PySDM.physics import constants as const
from ._gravitational import Gravitational


class Geometric(Gravitational):

    def __init__(self, collection_efficiency=1, x='volume'):
        super().__init__()
        self.collection_efficiency = collection_efficiency
        self.x = x
        self.collection_efficiency = 1

    def __call__(self, output, is_first_in_pair):
        output.sum(self.core.particles['radius'], is_first_in_pair)
        output **= 2
        output *= const.pi * self.collection_efficiency
        self.tmp.distance(self.core.particles['terminal velocity'], is_first_in_pair)
        output *= self.tmp
