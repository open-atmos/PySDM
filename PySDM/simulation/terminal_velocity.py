"""
Created at 24.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.physics import constants as const


class TerminalVelocity:

    def __init__(self, particles):
        self.__values = particles.backend.array(particles.n_sd, dtype=float)
        self.particles = particles
        self.compute = self.Rogers_and_Yau_eq_8_5

    @property
    def values(self):
        self.compute()
        return self.__values

    def Rogers_and_Yau_eq_8_5(self):
        k = 1.19e6 / const.si.centimetre / const.si.second

        backend = self.particles.backend
        volume = self.particles.state.get_backend_storage('volume')
        backend.multiply(self.__values, volume, 3 / 4 / const.pi * k**(3/2))
        backend.power(self.__values, (2 / 3))
