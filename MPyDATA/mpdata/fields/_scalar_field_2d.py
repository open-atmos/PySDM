"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


import numpy as np
import numba


@numba.jitclass([
    ('halo', numba.int64),
    ('shape', numba.int64[:]),
    ('data', numba.float64[:, :]),
    ('i', numba.int64),
    ('j', numba.int64),
    ('axis', numba.int64)])
class ScalarField2D:
    def __init__(self, data, halo):
        self.halo = halo
        self.shape = np.zeros(2, dtype=np.int64)
        self.shape[0] = data.shape[0] + 2 * halo
        self.shape[1] = data.shape[1] + 2 * halo

        shape_without_halo = data.shape
        self.data = np.zeros((shape_without_halo[0] + 2 * halo, shape_without_halo[1] + 2 * halo), dtype=np.float64)
        self.data[halo:self.shape[0] - halo, halo:self.shape[1] - halo] = data[:, :]

        self.i = 0
        self.j = 0
        self.axis = 0

    def focus(self, i, j):
        self.i = i + self.halo
        self.j = j + self.halo

    def set_axis(self, axis):
        self.axis = axis

    def swap_memory(self, other):
        self.data, other.data = other.data, self.data

    def at(self, arg1, arg2):
        if self.axis == 1:
            return self.data[self.i + arg2, self.j + arg1]
        else:
            return self.data[self.i + arg1, self.j + arg2]

    def apply(self, function, arg1, arg2):
        for i in range(self.shape[0] - 2 * self.halo):
            for j in range(self.shape[1] - 2 * self.halo):
                self.focus(i, j)
                arg1.focus(i, j)
                arg2.focus(i, j)

                self.data[self.i, self.j] = 0
                for dim in range(2):
                    self.set_axis(dim)
                    arg1.set_axis(dim)
                    arg2.set_axis(dim)

                    self.data[self.i, self.j] += function(arg1, arg2)

    def get(self):
        results = self.data[
            self.halo: self.data.shape[0] - self.halo,
            self.halo: self.data.shape[1] - self.halo
        ]
        return results

    def fill_halos(self):
        # TODO: use set_axis and loop over dimensions
        # TODO: hardcoded periodic boundary
        self.data[: self.halo, :] = self.data[-2*self.halo:-self.halo, :]
        self.data[-self.halo:, :] = self.data[self.halo:2*self.halo, :]

        self.data[:, : self.halo] = self.data[:, -2 * self.halo:-self.halo]
        self.data[:, -self.halo:] = self.data[:, self.halo:2 * self.halo]
