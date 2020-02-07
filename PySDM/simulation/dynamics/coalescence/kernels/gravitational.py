"""
Created at 24.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.physics import constants as const


class Gravitational:
    # TODO: handle collection_efficiency
    def __init__(self, collection_efficiency=1, x='volume'):
        self.collection_efficiency = 1
        self.x = x

        self.__particles = None
        self.__tmp = None

    @property
    def particles(self):
        return self.__particles

    @particles.setter
    def particles(self, particles):
        self.__particles = particles
        self.__tmp = particles.backend.array(particles.n_sd, dtype=float)

    def __call__(self, output, is_first_in_pair):
        backend = self.__particles.backend
        x = self.particles.state.get_backend_storage(self.x)

        backend.multiply(self.__tmp, x, (3 / 4 / const.pi))
        backend.power(self.__tmp, (1 / 3))
        backend.sum_pair(output, self.__tmp, is_first_in_pair, self.particles.state._State__idx, self.particles.state.SD_num)
        backend.power(output, 2)
        backend.multiply_in_place(output, const.pi * self.collection_efficiency)

        backend.distance_pair(self.__tmp, self.particles.terminal_velocity.values, is_first_in_pair,
                              self.particles.state._State__idx, self.particles.state.SD_num)
        backend.multiply_in_place(output, self.__tmp)

    # TODO: cleanup
    def analytic_solution(self, x, t, x_0, N_0):
        return x * 0
