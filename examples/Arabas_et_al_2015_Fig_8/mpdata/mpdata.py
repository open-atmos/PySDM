"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import ScalarField, VectorField
from examples.Arabas_et_al_2015_Fig_8.mpdata.formulae import Formulae


# @numba.jitclass()
class MPDATA:
    def __init__(self, state: ScalarField, courant_field: VectorField, n_iters: int, halo: int):
        self.new = state
        self.old = ScalarField.copy(state)

        self.C_physical = courant_field
        self.C_antidiff = courant_field.clone()
        self.flux = courant_field.clone()

        self.n_iters = n_iters
        self.halo = halo

    # @numba.jit()
    def step(self):
        for i in range(self.n_iters):
            if i == 0:
                apply(function=Formulae.flux, args=(self.C_physical, self.old), out=self.flux, halo=self.halo)
                apply(function=Formulae.upwind, args=(self.flux, self.old), out=self.new, halo=self.halo)
            else:
                raise NotImplementedError()
                #apply(..., antidiff, in=self.state[key], out=self.C_antidiff[key])
                #apply(..., upwind, self.C_antidiff)
            self.new.swap_memory(self.old)


@numba.jit()
def apply(function, args: tuple, out: ScalarField, halo: int):
    for i in range(halo, out.shape[0] - halo):
        for j in range(halo, out.shape[1] - halo):
            out.focus(i, j)
            for arg in args:
                arg.focus(i, j)

            out.tmp[i, j] = 0
            for dim in (0, 1):  # TODO: from shape of out?
                out.set_axis(dim)
                for arg in args:
                    arg.set_axis(dim)

                out.tmp[i, j] += function(*args)
