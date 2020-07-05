"""
Created at 05.07.2020
"""

from ._gravitational import Gravitational
from PySDM.physics import constants as const


class Parameterized(Gravitational):

    def __init__(self, params):
        super().__init__()
        self.params = params

    def __call__(self, output, is_first_in_pair):
        self.tmp.sort_pair(self.particles.state['radius'], is_first_in_pair)
        self.particles.backend.linear_collection_efficiency(
            self.params, output, self.tmp, is_first_in_pair, const.si.um)
        output **= 2
        output *= const.pi
        self.tmp **= 2
        output *= self.tmp

        self.tmp.distance_pair(self.particles.state['terminal velocity'], is_first_in_pair)
        output *= self.tmp