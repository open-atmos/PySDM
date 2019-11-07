"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


import numpy as np
import numba


@numba.jitclass([
    ('halo', numba.int64),
    ('shape_0', numba.int64),
    ('data', numba.float64[:]),
    ('i', numba.int64)])
class ScalarField1D:
    def __init__(self, data, halo):
        self.halo = halo
        self.shape_0 = data.shape[0]

        shape_with_halo = data.shape[0] + 2 * halo
        self.data = np.zeros(shape_with_halo, dtype=np.float64)
        self.data[halo:shape_with_halo - halo] = data[:]

        self.i = 0

    def focus(self, i):
        self.i = i + self.halo

    def swap_memory(self, other):
        self.data, other.data = other.data, self.data

    def at(self, item):
        return self.data[self.i + item]

    def apply(self, function, arg):
        for i in range(self.shape[0] - 2 * self.halo):
            self.focus(i)
            arg.focus(i)
            self.data[self.i] = function(arg)

    def get(self):
        results = self.data[self.halo: self.data.shape[0] - self.halo]
        return results

    def fill_halos(self):
        raise NotImplementedError()
