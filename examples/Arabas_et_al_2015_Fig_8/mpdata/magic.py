"""
Created at 27.09.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import ScalarField, VectorField


@numba.jit()
def f(psi, C):
    outflow = C.ij(0, 0) * psi.ij(0, 0)
    inflow = C.ij(0, 0) * psi.ij(-1, 0)
    return inflow - outflow


@numba.jit()
def apply(grid, function, psi, C):
    for i in range(1, grid[0] + 1):
        for j in range(1, grid[1] + 1):
            psi.focus(i, j)
            C.focus(i, j)

            # X
            tmp = function(psi, C)
            psi.swap_axis()
            C.swap_axis()

            # Y
            tmp += function(psi, C)
            psi.swap_axis()
            C.swap_axis()

            psi.tmp[i, j] = tmp + psi.data[i, j]
    psi.swap_memory()


if __name__ == '__main__':
    grid = (4, 4)
    init_psi = np.zeros((grid[0] + 2, grid[1] + 2), dtype=np.float32)
    init_psi[1, 1] = 1.

    psi = ScalarField(init_psi)
    C = VectorField(np.ones((grid[0] + 2, grid[1] + 2), dtype=np.float32) * .5)

    for _ in range(3):  # time-stepping
        apply(grid, f, psi, C)
        print(psi.data)
