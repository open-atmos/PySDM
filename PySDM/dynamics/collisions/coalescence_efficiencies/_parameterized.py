"""
Created 24.05.2021 by edejong
"""

from PySDM.physics import constants as const


class Parameterized():

    def __init__(self, params):
        self.particulator = None
        self.params = params

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute('radius')

    def __call__(self, output, is_first_in_pair):
        self.particulator.backend.linear_collection_efficiency(
            self.params, output, self.particulator.particles['radius'],
            is_first_in_pair, const.si.um)
        output **= 2
