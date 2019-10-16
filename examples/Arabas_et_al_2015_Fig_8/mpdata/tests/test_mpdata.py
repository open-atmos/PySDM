"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from examples.Arabas_et_al_2015_Fig_8.mpdata.mpdata import MPDATA
from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import ScalarField, VectorField
import numpy as np


class TestMPDATA:

    def test(self):
        scalar_field_init = np.array([[0., 1., 0.]]).T
        vector_field_init_x = np.full((4, 1), 1.)
        vector_field_init_y = np.full((3, 2), 0.)
        halo = 1
        scalar_field = ScalarField(scalar_field_init, halo=halo)
        vector_field = VectorField((vector_field_init_x, vector_field_init_y), halo=halo)

        mpdata = MPDATA(courant_field=vector_field, state=scalar_field, halo=halo, n_iters=1)
        print(mpdata)
        mpdata.step()