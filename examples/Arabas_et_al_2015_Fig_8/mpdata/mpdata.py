"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
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
        self.old = ScalarField.clone(state)

        self.C_physical = courant_field
        self.C_antidiff = courant_field.clone()
        self.flux = courant_field.clone()

        self.n_iters = n_iters
        self.halo = halo

    # @numba.jit()
    def step(self):
        for i in range(self.n_iters):
            if i == 0:
                self.flux.apply(function=Formulae.flux, args=(self.old, self.C_physical), halo=self.halo)
                self.new.apply(function=Formulae.upwind, args=(self.flux,), halo=self.halo)
                self.new.data += self.old.data
            else:
                raise NotImplementedError()
                #apply(..., antidiff, in=self.state[key], out=self.C_antidiff[key])
                #apply(..., upwind, self.C_antidiff)
            self.new.swap_memory(self.old)

