"""
Created at 24.01.2020
"""

from PySDM.physics import constants as const


class Gravitational:
    def __init__(self, collection_efficiency=1, x='volume'):
        self.collection_efficiency = collection_efficiency
        self.x = x

        self.particles = None
        self.__tmp = None
        if collection_efficiency == "hydrodynamic capture":
            self.params = (1, 1, -27, 1.65, -58, 1.9, 15, 1.13, 16.7, 1, .004, 4, 8)
            self.call = Gravitational.linear_collection_efficiency
        elif collection_efficiency == "electric field 3000V/cm":
            self.params = (1, 1, -7, 1.78, -20.5, 1.73, .26, 1.47, 1, .82, -0.003, 4.4, 8)
            self.call = Gravitational.linear_collection_efficiency
        elif collection_efficiency == "geometric sweep-out":
            self.collection_efficiency = 1
            self.call = Gravitational.collection_efficiency
        else:
            self.call = Gravitational.collection_efficiency

    def __call__(self, output, is_first_in_pair):
        self.call(self, output, is_first_in_pair)

    def register(self, particles_builder):
        self.particles = particles_builder.particles
        particles_builder.request_attribute('radius')
        particles_builder.request_attribute('terminal velocity')
        self.__tmp = self.particles.backend.IndexedStorage.empty(self.particles.n_sd, dtype=float)

    def collection_efficiency(self, output, is_first_in_pair):
        output.sum_pair(self.particles.state['radius'], is_first_in_pair)
        output **= 2
        output *= const.pi * self.collection_efficiency
        self.__tmp.distance_pair(self.particles.state['terminal velocity'], is_first_in_pair)
        output *= self.__tmp

    def linear_collection_efficiency(self, output, is_first_in_pair):
        self.__tmp.sort_pair(self.particles.state['radius'], is_first_in_pair)
        self.particles.backend.linear_collection_efficiency(
            self.params, output, self.__tmp, is_first_in_pair, const.si.um)
        output **= 2
        output *= const.pi
        self.__tmp **= 2
        output *= self.__tmp

        self.__tmp.distance_pair(self.particles.state['terminal velocity'], is_first_in_pair)
        output *= self.__tmp
