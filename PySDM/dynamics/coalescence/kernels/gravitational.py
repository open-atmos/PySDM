"""
Created at 24.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.physics import constants as const


class Gravitational:
    def __init__(self, collection_efficiency=1, x='volume'):
        self.collection_efficiency = collection_efficiency
        self.x = x

        self.particles = None
        self.__tmp = None
        if collection_efficiency == "hydrodynamic":
            self.params = (1, 1, -27, 1.65, -58, 1.9, 15, 1.13, 16.7, 1, .004, 4, 8)
            self.call = Gravitational.linear_collection_efficiency
        elif collection_efficiency == "3000V/cm":
            self.params = (1, 1, -7, 1.78, -20.5, 1.73, .26, 1.47, 1, .82, -0.003, 4.4, 8)
            self.call = Gravitational.linear_collection_efficiency
        else:
            self.call = Gravitational.collection_efficiency

    def __call__(self, output, is_first_in_pair):
        self.call(self, output, is_first_in_pair)

    def register(self, particles_builder):
        self.particles = particles_builder.particles
        particles_builder.request_attribute('radius')
        particles_builder.request_attribute('terminal velocity')
        self.__tmp = self.particles.backend.array(self.particles.n_sd, dtype=float)

    def collection_efficiency(self, output, is_first_in_pair):
        backend = self.particles.backend
        self.particles.sum_pair(output, 'radius', is_first_in_pair)
        backend.power(output, 2)
        backend.multiply(output, const.pi * self.collection_efficiency)

        backend.distance_pair(self.__tmp, self.particles.state['terminal velocity'], is_first_in_pair,
                              self.particles.state._State__idx, self.particles.state.SD_num)
        backend.multiply(output, self.__tmp)

    def linear_collection_efficiency(self, output, is_first_in_pair):
        idx = self.particles.state._State__idx
        length = self.particles.state.SD_num
        backend = self.particles.backend

        backend.sort_pair(self.__tmp, self.particles.state['radius'], is_first_in_pair, idx, length)
        backend.linear_collection_efficiency(self.params, output, self.__tmp, is_first_in_pair, length, const.si.um)
        backend.power(output, 2)
        backend.multiply(output, const.pi)
        backend.power(self.__tmp, 2)
        backend.multiply(output, self.__tmp)

        backend.distance_pair(self.__tmp, self.particles.state['terminal velocity'], is_first_in_pair, idx, length)
        backend.multiply(output, self.__tmp)

    # TODO: cleanup
    def analytic_solution(self, x, t, x_0, N_0):
        return x * 0
