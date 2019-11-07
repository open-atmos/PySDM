"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from MPyDATA.mpdata.fields.utils import is_integral


@numba.jitclass([
    ('halo', numba.int64),
    ('shape_0', numba.int64),
    ('data_0', numba.float64[:]),
    ('i', numba.int64)])
class VectorField1D:
    def __init__(self, data_0, halo):
        assert halo > 0
        self.halo = halo
        self.shape_0 = data_0.shape[0]
        self.data_0 = np.zeros((data_0.shape[0] + 2 * (halo - 1)), dtype=np.float64)

        shape_with_halo = data_0.shape[0] + 2 * (halo - 1)
        self.data_0[halo - 1:shape_with_halo - (halo - 1)] = data_0[:]

        self.i = 0

    @property
    def dimension(self):
        return 1

    def focus(self, i):
        self.i = i + self.halo - 1

    def at(self, item):
        idx = self.__idx(item)
        return self.data_0[idx]

    def __idx(self, item):
        if is_integral(item):
            raise ValueError()
        return self.i + int(item + .5)

    def get_component(self):
        return self.data_0[self.halo - 1: self.data_0.shape[0] - self.halo + 1]

    def apply(self, function, arg_1, arg_2):
        for i in range(-1, self.shape[0]):
            self.focus(i)
            arg_1.focus(i)
            arg_2.focus(i)

            idx = self.__idx(+.5)
            self.data_0[idx] = 0
            self.data_0[idx] += function(arg_1, arg_2)
