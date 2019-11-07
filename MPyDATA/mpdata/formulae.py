"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.mpdata.fields.scalar_field import ScalarField
from MPyDATA.mpdata.fields.vector_field import VectorField
import numpy as np
import numba


class Formulae:
    EPS = 1e-8
    HALO = 1

    @staticmethod
    @numba.njit()
    def flux(psi: ScalarField, C: VectorField):
        result = (
                np.maximum(0, C.at(+.5, 0)) * psi.at(0, 0) +  # TODO
                np.minimum(0, C.at(+.5, 0)) * psi.at(1, 0)
        )
        return result
        # TODO: check if (abs(c)-C)/2 is not faster

    @staticmethod
    @numba.njit()
    def upwind(flx: VectorField):
        return - (
                flx.at(+.5, 0) -
                flx.at(-.5, 0)
        )

    # TODO comment
    @staticmethod
    def A(psi: ScalarField):
        result = psi.at(1, 0) - psi.at(0, 0)
        result /= (psi.at(1, 0) + psi.at(0, 0) + Formulae.EPS)

        return result

    @staticmethod
    def antidiff(psi: ScalarField, C: VectorField):
        result = (np.abs(C.at(+.5, 0)) - C.at(+.5, 0) ** 2) * Formulae.A(psi)

        for i in range(len(psi.shape)):
            if i == psi.axis:
                continue
            result -= 0.5 * C.at(+.5, 0) * 0.25 * (C.at(1, +.5) + C.at(0, +.5) + C.at(1, -.5) + C.at(0, -.5)) * \
                      (psi.at(1, 1) + psi.at(0, 1) - psi.at(1, -1) - psi.at(0, -1)) / \
                      (psi.at(1, 1) + psi.at(0, 1) + psi.at(1, -1) + psi.at(0, -1) + Formulae.EPS)
            # TODO dx, dt

        return result




