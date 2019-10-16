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
        assert state.data.shape[0] == courant_field.data[0].shape[0] + 1
        assert state.data.shape[1] == courant_field.data[0].shape[1] + 2
        assert courant_field.data[0].shape[0] == courant_field.data[1].shape[0] + 1
        assert courant_field.data[0].shape[1] == courant_field.data[1].shape[1] - 1

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

    def __str__(self):
        print()
        color = '\033[94m'
        bold = '\033[7m'
        endcolor = '\033[0m'

        for i in range(self.new.data.shape[0]):
            # i,j
            for j in range(self.new.data.shape[1]):
                is_scalar_halo = (
                        i < self.halo or
                        j < self.halo or
                        i >= self.new.data.shape[0] - self.halo or
                        j >= self.new.data.shape[1] - self.halo
                )
                is_not_vector_halo = True #(  # TODO: halo>1
                        # self.halo - 1 <  i < self.new.data.shape[0] - self.halo and
                        # self.halo - 1 <= j < self.new.data.shape[1] - self.halo
                # )

                if is_scalar_halo:
                    print(color, end='')
                else:
                    print(bold, end='')
                svalue = '{:04.1f}'.format(self.new.ij(i,j))
                print(f"\t({i},{j})={svalue}", end = endcolor)

                # i+.5,j
                if (is_not_vector_halo):
                    vvalue = '{:04.1f}'.format(self.C_physical.at(i, j - .5))
                    print(f'\t({i},{j-.5})={vvalue}', end='')
                else:
                    print('\t' * 4, end='')

            print('')
            for j in range(-self.halo, self.new.data.shape[1]):
                if j == 1 and i <= 3: # TODO
                    vvalue = '{:04.1f}'.format(self.C_physical.at(i-.5, j))
                    print(f"\t({i-.5},{j})={vvalue}", end='')
                else:
                    print("\t" * 7, end='')
            print('')

