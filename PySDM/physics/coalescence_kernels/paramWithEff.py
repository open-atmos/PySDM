"""
Created at 18.05.2021
"""

from ._gravitational import Gravitational
from PySDM.physics import constants as const


class ParamWithEff(Gravitational):

    def __init__(self):
        super().__init__()

    def __call__(self, output, is_first_in_pair):
        output.max(self.core.particles['radius'], is_first_in_pair)
        output **= 2
        output *= const.pi

        self.pair_tmp.distance(self.core.particles['terminal velocity'], is_first_in_pair)
        output *= self.pair_tmp
