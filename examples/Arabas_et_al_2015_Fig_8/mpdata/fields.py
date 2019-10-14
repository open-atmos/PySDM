"""
Created at 02.10.2019

@author: Piotr Bartman
"""

import numpy as np
import numba


# @numba.jitclass([
#     ('shape', numba.int32[:]),
#     ('data', numba.float32[:, :]),
#     ('i', numba.int32),
#     ('j', numba.int32),
#     ('transpose', numba.boolean)])
class ScalarField:
    def __init__(self, data, halo):
        self.shape = (data.shape[0] + 2*halo, data.shape[1] + 2*halo)
        self.data = np.zeros(self.shape, dtype=np.float32)
        self.data[halo:self.shape[0] - halo, halo:self.shape[1] - halo] = data[:, :]

        self.i = 0
        self.j = 0
        self.transpose = False

    def focus(self, i, j):
        self.i = i
        self.j = j

    def swap_axis(self):
        self.transpose = not self.transpose

    def swap_memory(self, other):
        self.data, other.data = other.data, self.data

    def ij(self, arg1, arg2):
        if self.transpose:
            return self.data[self.i + arg2, self.j + arg1]
        else:
            return self.data[self.i + arg1, self.j + arg2]

    # TODO !!!
    def copy(self):
        return ScalarField(data=self.data, halo=0)


# @numba.jitclass([
#     ('data1', numba.float32[:, :]),
#     ('data2', numba.float32[:, :]),
#     ('i', numba.int32),
#     ('j', numba.int32),
#     ('transpose', numba.boolean)])
class VectorField:
    def __init__(self, data, halo):
        self.data = []
        for d in data:
            shape = tuple([d.shape[i] + 2 * (halo - 1) for i in range(len(d.shape))])
            self.data.append(np.zeros(shape, dtype=np.float32))
            ijk = tuple([slice(halo - 1, shape[i] - (halo - 1)) for i in range(len(d.shape))])
            self.data[-1][ijk] = d

        # TODO probably not working with numba
        self.i = 0
        if len(data.shape) > 1:
            self.j = 0
            self.axis = 0
        if len(data.shape) > 2:
            self.k = 0
        if len(data.shape) > 3:
            raise NotImplementedError()

    def focus(self, i, j=0, k=0):
        self.i = i
        self.j = j
        self.k = k
        # TODO: depending on number of dims

    # TODO: set_dim
    def set_axis(self, axis):
        self.axis = axis

    def at(self, *args):
        if len(args) == 1:
            return self.at_1d(*args)
        elif len(args) == 2:
            return self.at_2d(*args)
        else:
            raise NotImplementedError()

    def at_1d(self, arg1):
        idx1 = int(arg1 + .5)
        assert idx1 == arg1 + .5
        return self.data[self.i + idx1]

    def at_2d(self, arg1, arg2):
        if arg1 is int and arg2 is float:
            data = self.data[1]
            idx1 = arg1
            idx2 = int(arg2 + .5)
            assert idx2 == arg2 + .5
        elif arg2 is int and arg1 is float:
            data = self.data[0]
            idx1 = int(arg1 + .5)
            idx2 = arg2
            assert idx1 == arg1 + .5
        else:
            raise NotImplementedError()

        if self.axis == 1:
            return data[self.i + idx2, self.j + idx1]
        else:
            return data[self.i + idx1, self.j + idx2]

    # TODO !!!
    def clone(self, value=np.nan):
        data1 = np.full_like(self.data1, value)
        data2 = np.full_like(self.data2, value)
        return VectorField(data1, data2, halo=1)



