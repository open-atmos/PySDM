"""
basic geometric kernel
"""
from PySDM.dynamics.collisions.collision_kernels.impl.gravitational import Gravitational
from PySDM.physics import constants as const


class Geometric(Gravitational):
    def __init__(self, collection_efficiency=1.0, x="volume", relax_velocity=False):
        super().__init__(relax_velocity=relax_velocity)
        self.collection_efficiency = collection_efficiency
        self.x = x

    def __call__(self, output, is_first_in_pair):
        output.sum(self.particulator.attributes["radius"], is_first_in_pair)
        output **= 2
        output *= const.PI * self.collection_efficiency
        self.pair_tmp.distance(
            self.particulator.attributes[self.vel_attr], is_first_in_pair
        )
        output *= self.pair_tmp
