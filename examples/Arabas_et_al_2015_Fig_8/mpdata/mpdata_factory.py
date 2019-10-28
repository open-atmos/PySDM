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
    def kinematic_2d(grid, size, dt, stream_function: callable, field_values: dict, halo=1):
        courant_field = nondivergent_vector_field_2d(grid, size, halo, dt, stream_function)

        mpdatas = {}
        for key, value in field_values.items():
            state = uniform_scalar_field(grid, value, halo)
            mpdatas[key] = MPDATAFactory.mpdata(state=state, courant_field=courant_field, n_iters=1)

        eulerian_fields = EulerianFields(mpdatas)
        return courant_field, eulerian_fields


def uniform_scalar_field(grid, value, halo):
    data = np.full(grid, value)
    scalar_field = ScalarField(data=data, halo=halo)
    return scalar_field


def nondivergent_vector_field_2d(grid, size, halo, dt, stream_function: callable):
    # TODO: density!

    dx = size[0] / grid[0]
    dz = size[1] / grid[1]

    nx = grid[0]+1
    nz = grid[1]
    x = np.repeat(np.linspace(0, size[0], nx).reshape((nx, 1)), nz, axis=1)
    assert x.shape == (nx, nz)
    z = np.repeat(np.linspace(dz / 2, size[1] - dz/2, nz).reshape((1, nz)), nx, axis=0)
    assert z.shape == (nx, nz)
    velocity_x = -(stream_function(x, z + dz / 2) - stream_function(x, z - dz / 2)) / dz

    nx = grid[0]
    nz = grid[1]+1
    x = np.repeat(np.linspace(dx/2, size[0]-dx/2, nx).reshape((nx, 1)), nz, axis=1)
    assert x.shape == (nx, nz)
    z = np.repeat(np.linspace(0, size[1], nz).reshape((1, nz)), nx, axis=0)
    assert z.shape == (nx, nz)
    velocity_z = (stream_function(x + dx / 2, z) - stream_function(x - dx / 2, z)) / dx

    courant_field = [velocity_x * dt / dx, velocity_z * dt / dz]
    print(courant_field[0])
    for d in range(len(courant_field)):
        np.testing.assert_array_less(np.abs(courant_field[d]), 1)

    return VectorField(data=courant_field, halo=halo)
