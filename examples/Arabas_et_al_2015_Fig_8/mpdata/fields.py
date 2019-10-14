"""
Created at 02.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
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
        self.shape = (data.shape[0] + 2 * halo, data.shape[1] + 2 * halo)
        self.data = np.zeros(self.shape, dtype=np.float32)
        self.data[halo:self.shape[0] - halo, halo:self.shape[1] - halo] = data[:, :]

        self.i = 0
        self.j = 0
        self.axis = 0

    def focus(self, i, j):
        self.i = i
        self.j = j

    def set_axis(self, axis):
        self.axis = axis

    def swap_memory(self, other):
        self.data, other.data = other.data, self.data

    def ij(self, arg1, arg2):
        if self.axis == 1:
            return self.data[self.i + arg2, self.j + arg1]
        else:
            return self.data[self.i + arg1, self.j + arg2]

    # TODO !!!
    def clone(self):
        return ScalarField(data=self.data, halo=0)

    def apply(self, function, args: tuple, halo: int):
        print(self.shape)
        for i in range(halo, self.shape[0] - halo):
            for j in range(halo, self.shape[1] - halo):
                self.focus(i, j)
                print(i, j)
                for arg in args:
                    arg.focus(i, j)

                self.data[i, j] = 0
                for dim in (0, 1):  # TODO: from shape?
                    self.set_axis(dim)
                    for arg in args:
                        arg.set_axis(dim)

                    self.data[i, j] += function(*args)


# @numba.jitclass([
#     ('data1', numba.float32[:, :]),
#     ('data2', numba.float32[:, :]),
#     ('i', numba.int32),
#     ('j', numba.int32),
#     ('transpose', numba.boolean)])
class VectorField:
    def __init__(self, data, halo):
        assert halo > 0
        if len(data) == 2:
            assert data[0].shape[0] == data[1].shape[0] + 1
            assert data[0].shape[1] == data[1].shape[1] - 1
        self.data = []
        for d in data:
            shape = tuple([d.shape[i] + 2 * (halo - 1) for i in range(len(d.shape))])
            self.data.append(np.zeros(shape, dtype=np.float32))
            ijk = tuple([slice(halo - 1, shape[i] - (halo - 1)) for i in range(len(d.shape))])
            self.data[-1][ijk] = d

        # TODO probably not working with numba
        self.i = 0
        if len(data) > 1:
            self.j = 0
            self.axis = 0
        if len(data) > 2:
            self.k = 0
        if len(data) > 3:
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
        return self.data[0][self.i + idx1]

    def at_2d(self, arg1, arg2):
        if self.axis == 1:
            arg1, arg2 = arg2, arg1

        if isinstance(arg1, int) and isinstance(arg2, float):
            data = self.data[1]
            idx1 = arg1 - 1
            idx2 = int(arg2 - .5)
            assert idx2 == arg2 - .5
        elif isinstance(arg2, int) and isinstance(arg1, float):
            data = self.data[0]
            idx1 = int(arg1 - .5)
            idx2 = arg2 - 1
            assert idx1 == arg1 - .5
        else:
            raise NotImplementedError()

        return data[self.i + idx1, self.j + idx2]

    # TODO !!!
    def clone(self, value=np.nan):
        data1 = np.full_like(self.data[0], value)
        data2 = np.full_like(self.data[1], value)
        return VectorField([data1, data2], halo=1)

    def apply(self, function, args: tuple, halo: int):
        for d in range(2):
            for i in range(halo, self.data[d].shape[0] - halo):
                for j in range(halo, self.data[d].shape[1] - halo):
                    self.focus(i, j)
                    for arg in args:
                        arg.focus(i, j)

                    self.data[d][i, j] = 0
                    for dim in (0, 1):  # TODO: from shape of out?
                        self.set_axis(dim)
                        for arg in args:
                            arg.set_axis(dim)

                        self.data[d][i, j] += function(*args)



