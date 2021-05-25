from ._gravitational import Gravitational
from PySDM.physics import constants as const


class Parameterized(Gravitational):

    def __init__(self, params):
        super().__init__()
        self.params = params

    def __call__(self, output, is_first_in_pair):
        self.core.backend.linear_collection_efficiency(
            self.params, output, self.core.particles['radius'], is_first_in_pair, const.si.um)
        output **= 2
        output *= const.pi
        self.pair_tmp.max(self.core.particles['radius'], is_first_in_pair)
        self.pair_tmp **= 2
        output *= self.pair_tmp

        self.pair_tmp.distance(self.core.particles['terminal velocity'], is_first_in_pair)
        output *= self.pair_tmp
