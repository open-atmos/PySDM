""" common parent class for collision kernels specified using Berry's parameterization """


from PySDM.physics import constants as const

from .gravitational import Gravitational


class Parameterized(Gravitational):
    def __init__(self, params, relax_velocity=False):
        super().__init__(relax_velocity=relax_velocity)
        self.params = params

    def __call__(self, output, is_first_in_pair):
        self.particulator.backend.linear_collection_efficiency(
            params=self.params,
            output=output,
            radii=self.particulator.attributes["radius"],
            is_first_in_pair=is_first_in_pair,
            unit=const.si.um,
        )
        output **= 2
        output *= const.PI
        self.pair_tmp.max(self.particulator.attributes["radius"], is_first_in_pair)
        self.pair_tmp **= 2
        output *= self.pair_tmp

        self.pair_tmp.distance(
            self.particulator.attributes[self.vel_attr], is_first_in_pair
        )
        output *= self.pair_tmp
