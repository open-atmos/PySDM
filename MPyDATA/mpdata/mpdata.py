"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.mpdata.fields.scalar_field import ScalarField
from MPyDATA.mpdata.fields.vector_field import VectorField
from MPyDATA.mpdata.formulae import Formulae


# @numba.jitclass()
class MPDATA:
    def __init__(self, prev: ScalarField, curr: ScalarField, G: ScalarField,
                 GC_physical: VectorField, GC_antidiff: VectorField,
                 flux: VectorField, n_iters: int, halo: int):
        self.curr = curr
        self.prev = prev
        self.G = G
        self.GC_physical = GC_physical
        self.GC_antidiff = GC_antidiff
        self.flux = flux

        self.n_iters = n_iters
        self.halo = halo

    # @numba.jit()
    def step(self):
        for i in range(self.n_iters):
            self.prev.swap_memory(self.curr)
            self.prev.fill_halos()
            if i == 0:
                GC = self.GC_physical
            else:
                self.GC_antidiff.apply(Formulae.antidiff, self.prev, self.GC_physical)
                GC = self.GC_antidiff
            self.flux.apply(Formulae.flux, self.prev, GC)
            self.curr.apply(Formulae.upwind, self.flux, self.G)
            self.curr.data += self.prev.data

    def debug_print(self):
        print()
        color = '\033[94m'
        bold = '\033[7m'
        endcolor = '\033[0m'

        shp0 = self.curr.data.shape[0]
        shp1 = self.curr.data.shape[1]

        self.GC_physical.focus(0,0)
        self.GC_physical.set_axis(0)
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
                svalue = '{:04.1f}'.format(self.curr.at(i,j))
                print(f"\t{svalue}", end = endcolor)

                # i+.5,j
                if (is_not_vector_halo):
                    vvalue = '{:04.1f}'.format(self.GC_physical.at(i, j + .5))
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
                    vvalue = '{:04.1f}'.format(self.GC_physical.at(i + .5, j))
                    print(f"\t\t\t{vvalue}", end='')
                else:
                    print("\t" * 2, end='')
            print('')

