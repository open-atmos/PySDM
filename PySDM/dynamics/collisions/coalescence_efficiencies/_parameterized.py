"""
TODO #744
"""

from PySDM.physics import constants as const


class Parameterized:
    def __init__(self, params):
        self.particulator = None
        self.params = params

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("radius")

    def __call__(self, output, is_first_in_pair):
        self.particulator.backend.linear_collection_efficiency(
            params=self.params,
            output=output,
            radii=self.particulator.attributes["radius"],
            is_first_in_pair=is_first_in_pair,
            unit=const.si.um,
        )
        output **= 2
