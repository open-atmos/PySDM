"""
Created at 31.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numba

from SDM.backends.numba import Numba


class NumbaParallel(Numba):
    @staticmethod
    @numba.njit("void(int32[:], int32[:], int32, float64[:,:], float64[:])", parallel=True)
    def extensive_attr_coalescence(n, idx, length, data, gamma):
        # TODO in segments
        for i in numba.prange(length // 2):
            j = 2 * i
            k = j + 1

            j = idx[j]
            k = idx[k]

            if n[j] < n[k]:
                j, k = k, j
            g = min(gamma[i], n[j] // n[k])

            new_n = n[j] - g * n[k]
            if new_n > 0:
                data[:, k] += g * data[:, j]
            else:  # new_n == 0
                data[:, j] = g * data[:, j] + data[:, k]
                data[:, k] = data[:, j]