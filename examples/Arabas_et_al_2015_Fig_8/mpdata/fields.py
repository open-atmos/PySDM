"""
Created at 02.10.2019

@author: Piotr Bartman
"""

import numpy as np
import numba


@numba.jitclass([
    ('data', numba.float32[:, :]),
    ('tmp', numba.float32[:, :]),
    ('i', numba.int32),
    ('j', numba.int32),
    ('transpose', numba.boolean)])
class ScalarField:
    def __init__(self, data, halo):
        shape = (data.shape[0] + halo, data.shape[1] + halo)
        self.data = np.zeros(shape, dtype=np.float32)
        self.data[halo:-halo, halo:-halo] = data[:, :]
        self.tmp = np.zeros(shape, dtype=np.float32)

        self.i = 0
        self.j = 0
        self.transpose = False

    def focus(self, i, j):
        self.i = i
        self.j = j

    def swap_axis(self):
        self.transpose = not self.transpose

    def swap_memory(self):
        self.data, self.tmp = self.tmp, self.data

    def ij(self, arg1, arg2):
        if self.transpose:
            return self.data[self.i + arg2, self.j + arg1]
        else:
            return self.data[self.i + arg1, self.j + arg2]


@numba.jitclass([
    ('data', numba.float32[:, :]),
    ('tmp', numba.float32[:, :]),
    ('i', numba.int32),
    ('j', numba.int32),
    ('transpose', numba.boolean)])
class VectorField:
    def __init__(self, data1, data2, halo):
        shape = (data1.shape[0] + halo, data1.shape[1] + halo)
        self.data1 = np.zeros(shape, dtype=np.float32)
        self.data1[halo:-halo, halo:-halo] = data1[:, :]
        self.data2 = np.zeros(shape, dtype=np.float32)
        self.data2[halo:-halo, halo:-halo] = data2[:, :]

        self.i = 0
        self.j = 0
        self.transpose = False

    def focus(self, i, j):
        self.i = i
        self.j = j

    def swap_axis(self):
        self.transpose = not self.transpose

    def ij(self, arg1, arg2):
        # TODO assert?
        arg1 = int(arg1 - 0.5)
        arg2 = int(arg2 - 0.5)
        if self.transpose:
            return self.data[self.i + arg2, self.j + arg1]
        else:
            return self.data[self.i + arg1, self.j + arg2]