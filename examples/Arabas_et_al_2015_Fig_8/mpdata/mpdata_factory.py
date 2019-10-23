"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import ScalarField, VectorField
from examples.Arabas_et_al_2015_Fig_8.mpdata.mpdata import MPDATA
from examples.Arabas_et_al_2015_Fig_8.mpdata.eulerian_fields import EulerianFields


class MPDATAFactory:
    @staticmethod
    def mpdata(state: ScalarField, courant_field: VectorField, n_iters):
        assert state.data.shape[0] == courant_field.data[0].shape[0] + 1
        assert state.data.shape[1] == courant_field.data[0].shape[1] + 2
        assert courant_field.data[0].shape[0] == courant_field.data[1].shape[0] + 1
        assert courant_field.data[0].shape[1] == courant_field.data[1].shape[1] - 1
        # TODO assert halo

        prev = state.clone()
        C_antidiff = courant_field.clone()
        flux = courant_field.clone()
        halo = state.halo
        mpdata = MPDATA(curr=state, prev=prev, C_physical=courant_field, C_antidiff=C_antidiff, flux=flux,
                        n_iters=n_iters, halo=halo)

        return mpdata

    @staticmethod
    def kinematic_2d(grid, size, stream_function: callable, field_values: dict, halo=1):
        courant_field = nondivergent_vector_field(grid, size, halo, stream_function)

        mpdatas = {}
        for key, value in field_values.items():
            state = uniform_scalar_field(grid, value, halo)
            mpdatas[key] = MPDATAFactory.mpdata(state=state, courant_field=courant_field)

        eulerian_fields = EulerianFields(mpdatas)
        return eulerian_fields


def uniform_scalar_field(grid, value, halo):
    data = np.full(grid, value)
    scalar_field = ScalarField(data=data, halo=halo)
    return scalar_field


def nondivergent_vector_field_2d(grid, size, halo, stream_function: callable):
    dx = grid[0] / size[0]
    dz = grid[1] / size[1]
    data_x = np.empty((grid[0] + 1, grid[1]))
    data_z = np.empty((grid[0], grid[1] + 1))
    # TODO
    data = [data_x, data_z]
    vector_field = VectorField(data=data, halo=halo)
    return vector_field
