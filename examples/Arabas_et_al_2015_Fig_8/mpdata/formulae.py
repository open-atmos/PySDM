"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import ScalarField, VectorField
import numpy as np
import numba


class Formulae:
    @staticmethod
    # @numba.jit()
    def flux(psi: ScalarField, C: VectorField):
        return (
                np.maximum(0, C.at(-.5, 0)) * psi.ij(0, 0) +
                np.minimum(0, C.at(+.5, 0)) * psi.ij(1, 0)
        )
        # TODO: check if (abs(c)-C)/2 is not faster

    @staticmethod
    # @numba.jit()
    def upwind(flx: VectorField):
        return - (
                flx.at(+.5, 0) -
                flx.at(-.5, 0)
        )

