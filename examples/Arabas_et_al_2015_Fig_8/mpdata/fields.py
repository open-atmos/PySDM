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
        self.halo = halo
        self.shape = (data.shape[0] + 2 * halo, data.shape[1] + 2 * halo)  # TODO domain shape
        self.data = np.zeros(self.shape, dtype=np.float32)
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

    def ij(self, arg1, arg2):
        if self.axis == 1:
            return self.data[self.i + arg2, self.j + arg1]
        else:
            return self.data[self.i + arg1, self.j + arg2]

    # TODO: _2d, _1d, ...
    def clone(self):
        return ScalarField(data=self.data[
                                self.halo:-self.halo ,
                                self.halo:-self.halo ,
                                ], halo=self.halo)

    def apply(self, function, args: tuple):
        for i in range(self.shape[0] - 2 * self.halo):
            for j in range(self.shape[1] - 2 * self.halo):
                self.focus(i, j)
                for arg in args:
                    arg.focus(i, j)

                self.data[self.i, self.j] = 0
                for dim in (0, 1):  # TODO: from shape?
                    self.set_axis(dim)
                    for arg in args:
                        arg.set_axis(dim)

                    self.data[self.i, self.j] += function(*args)


# @numba.jitclass([
#     ('data1', numba.float32[:, :]),
#     ('data2', numba.float32[:, :]),
#     ('i', numba.int32),
#     ('j', numba.int32),
#     ('transpose', numba.boolean)])
class VectorField:
    def __init__(self, data, halo):
        assert halo > 0
        self.halo = halo
        if len(data) == 2:
            self.shape = (data[1].shape[0], data[0].shape[1])
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

    @property
    def dimension(self):
        return len(self.shape)

    def focus(self, i, j=0, k=0):
        self.i = i + self.halo - 1
        self.j = j + self.halo - 1
        self.k = k + self.halo - 1
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
        d, idx1, idx2 = self.__idx_2d(arg1, arg2)
        return self.data[d][idx1, idx2]

    def __idx_2d(self, arg1, arg2):
        if self.axis == 1:
            arg1, arg2 = arg2, arg1

        if isinstance(arg1, int) and isinstance(arg2, float):
            d = 1
            idx1 = arg1
            idx2 = int(arg2 + .5)
            assert idx2 == arg2 + .5
        elif isinstance(arg2, int) and isinstance(arg1, float):
            d = 0
            idx1 = int(arg1 + .5)
            idx2 = arg2
            assert idx1 == arg1 + .5
        else:
            raise NotImplementedError()

        assert self.i + idx1 >= 0
        assert self.j + idx2 >= 0

        return d, self.i + idx1, self.j + idx2

    def get_domain(self):
        pass

    def fill_halos(self):
        if self.boundary_cond == 'periodic':
            for d in range(self.dimension):
                self.set_axis(d)
                # self.data[left_halo, :] = self.data[right_edge, :]
                # self.data[right_halo, :] = self.data[left_edge, :]
        else:
            raise NotImplementedError()

    # TODO: _1d, _2d, ...
    def clone(self, value=np.nan):
        data1 = np.full_like(self.data[0], value)
        data2 = np.full_like(self.data[1], value)
        return VectorField([data1, data2], halo=1)  # TODO: !!! halo is now important!

    def apply(self, function, args: tuple):
        for i in range(-1, self.shape[0]):
            for j in range(-1, self.shape[1]):

                self.focus(i, j)
                for arg in args:
                    arg.focus(i, j)

                for dd in (0, 1):
                    if (i == -1 and dd == 1) or (j == -1 and dd == 0):
                        continue

                    self.set_axis(dd)
                    (d, ii, jj) = self.__idx_2d(+.5, 0)  # TODO !!!
                    self.data[d][ii, jj] = 0
                    for arg in args:
                        arg.set_axis(dd)

                    self.data[d][ii, jj] += function(*args)





