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

        self.call = Gravitational.collection_efficiency if collection_efficiency else Gravitational.linear_collection_efficiency

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
        backend = self.particles.backend
        x = self.particles.state[self.x]

        backend.multiply_out_of_place(self.__tmp, x, (3 / 4 / const.pi))
        backend.power(self.__tmp, (1 / 3))
        linear_collection_efficiency(output, self.__tmp, is_first_in_pair,
                                     self.particles.state._State__idx, self.particles.state.SD_num)
        backend.power(self.__tmp, 2)
        backend.multiply(output, const.pi)
        backend.multiply(output, self.__tmp)

        backend.distance_pair(self.__tmp, self.particles.terminal_velocity.values, is_first_in_pair,
                              self.particles.state._State__idx, self.particles.state.SD_num)
        backend.multiply(self.__tmp, output)
        sort(output, self.__tmp, self.particles.state._State__idx, self.particles.state.SD_num)

    # TODO: cleanup
    def analytic_solution(self, x, t, x_0, N_0):
        return x * 0

# TODO

import numba
from numba import  void, float64, int64, prange
from PySDM.backends.numba import conf

A = 1
B = 1
D1 = -27
D2 = 1.65
E1 = -58
E2 = 1.9
F1 = 15
F2 = 1.13
G1 = 16.7
G2 = 1
G3 = 0.004
Mf = 4
Mg = 8
um = const.si.um


@numba.njit()
def linear_collection_efficiency(output, radii, is_first_in_pair, idx, length):
    sort_pair(radii, is_first_in_pair, idx, length)
    for i in prange(length - 1):
        if is_first_in_pair[i]:
            output[idx[i]] = __linear_collection_efficiency(radii[idx[i]], radii[idx[i+1]])
        else:
            output[idx[i]] = 0



@numba.njit()
def __linear_collection_efficiency(r, r_s):
    r /= um
    r_s /= um
    p = r_s / r
    D = D1 / r ** D2
    E = E1/ r ** E2
    F = (F1 / r) ** Mf + F2
    G = (G1 / r) ** Mg + G2 + G3 * r
    if (1 - p) ** G == 0:  # TODO!
        result = 0
    else:
        result = A + B * p + D / p ** F + E / (1 - p) ** G
    return max(0, result)


@numba.njit(void(float64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
def sort_pair(data_in, is_first_in_pair, idx, length):
    for i in prange(length - 1):
        if is_first_in_pair[i] and data_in[idx[i]] < data_in[idx[i + 1]]:
            data_in[idx[i]], data_in[idx[i + 1]] = data_in[idx[i + 1]], data_in[idx[i]]

@numba.njit()
def sort(data_out, data_in, idx, length):
    for i in prange(length):
        data_out[i] = data_in[idx[i]]


@numba.njit(void(float64[:], float64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
def ratio_pair(data_out, data_in, is_first_in_pair, idx, length):
    for i in prange(length - 1):
        data_out[idx[i]] = (data_in[idx[i + 1]] / data_in[idx[i]]) if is_first_in_pair[i] else 0

