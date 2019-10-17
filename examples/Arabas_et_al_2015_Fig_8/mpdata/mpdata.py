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

        self.curr = state
        self.next = ScalarField.clone(state)  # TODO: prev?

        self.C_physical = courant_field
        self.C_antidiff = courant_field.clone()
        self.flux = courant_field.clone()

        self.n_iters = n_iters
        self.halo = halo

    # @numba.jit()
    def step(self):
        for i in range(self.n_iters):
            if i == 0:
                self.next.data[:] = 0
                self.flux.apply(function=Formulae.flux, args=(self.curr, self.C_physical))
                self.next.apply(function=Formulae.upwind, args=(self.flux,))
                self.next.data += self.curr.data
            else:
                raise NotImplementedError()
                #apply(..., antidiff, in=self.state[key], out=self.C_antidiff[key])
                #apply(..., upwind, self.C_antidiff)
            self.next.swap_memory(self.curr)

    def debug_print(self):
        print()
        color = '\033[94m'
        bold = '\033[7m'
        endcolor = '\033[0m'

        shp0 = self.curr.data.shape[0]
        shp1 = self.curr.data.shape[1]

        self.C_physical.focus(0,0)
        self.C_physical.set_axis(0)
        self.curr.focus(0,0)
        self.curr.set_axis(0)

        print("\t"*2, end='')
        for j in range(-self.halo, shp1 - self.halo):
            print("\t{:+.1f}".format(j), end='')
            if j != shp1-self.halo-1: print("\t{:+.1f}".format(j+.5), end='')
        print()

        for i in range(-self.halo, shp0-self.halo):
            print("\t{:+.1f}".format(i), end='')
            # i,j
            for j in range(-self.halo, shp1-self.halo):
                is_scalar_halo = (
                        i < 0 or
                        j < 0 or
                        i >= shp0-2*self.halo or
                        j >= shp1-2*self.halo
                )
                is_not_vector_halo = (
                    -(self.halo-1) <= i < shp0-2*(self.halo)+(self.halo-1) and
                        -self.halo <= j < shp1-2*(self.halo)+(self.halo-1)
                )

                if is_scalar_halo:
                    print(color, end='')
                else:
                    print(bold, end='')
                svalue = '{:04.1f}'.format(self.curr.ij(i,j))
                print(f"\t{svalue}", end = endcolor)

                # i+.5,j
                if (is_not_vector_halo):
                    vvalue = '{:04.1f}'.format(self.C_physical.at(i, j+.5))
                    print(f'\t{vvalue}', end='')
                else:
                    print('\t' * 2, end='')

            print('')
            if (i < shp0-(self.halo-1)-2):
                print("\t{:+.1f}".format(i+.5), end='')
            for j in range(-self.halo, shp1 - self.halo):
                pass
                if (
                    -(self.halo-1) <= j < shp1-(self.halo-1)-2 and
                    -self.halo <= i < shp0-(self.halo-1)-2
                ):
                    vvalue = '{:04.1f}'.format(self.C_physical.at(i+.5, j))
                    print(f"\t\t\t{vvalue}", end='')
                else:
                    print("\t" * 2, end='')
            print('')

