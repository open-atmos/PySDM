"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import ScalarField, VectorField


class MPDATA:
    def __init__(self, init_values: dict, courant_field: tuple, n_iters: int):
        # TODO check values shape
        self.halo = 1
        self.state = {}
        for key in init_values:
            self.state[key] = ScalarField(init_values[key], halo=self.halo)
        self.C_physical = VectorField(*courant_field, halo=self.halo)
        self.C_antidiff = VectorField(*courant_field, halo=self.halo)
        self.n_iters = n_iters

    def step(self):
        for i in range(self.n_iters):
            if i == 0:
                apply(..., upwind, self.C_physical)
            else:
                apply(..., antidiff, self.C_antidiff)
                apply(..., upwind, self.C_antidiff)

    @staticmethod
    def upwind(psi, flx, G, i):
        return psi[i] - (flx[i + HALF] - flx[i - HALF]) / G[i]


@numba.jit()
def apply(array, function, halo):
    for i in range(halo, array.shape[0] - halo):
        for j in range(halo, array.shape[0] - halo):
            array.focus(i, j)
            C.focus(i, j)

            # X
            tmp = function(psi, C)
            array.swap_axis()
            C.swap_axis()

            # Y
            tmp += function(psi, C)
            array.swap_axis()
            C.swap_axis()

            array.tmp[i, j] = tmp + array.data[i, j]
    array.swap_memory()