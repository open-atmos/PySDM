"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.mpdata.fields.scalar_field import ScalarField
from MPyDATA.mpdata.fields import scalar_field
from MPyDATA.mpdata.fields.vector_field import VectorField
from MPyDATA.mpdata.fields import vector_field
from MPyDATA.mpdata.mpdata import MPDATA
from MPyDATA.mpdata.eulerian_fields import EulerianFields


class MPDATAFactory:
    @staticmethod
    def mpdata(state: ScalarField, g_factor: ScalarField, GC_field: VectorField, n_iters):
        assert state.data.shape[0] == GC_field.data(0).shape[0] + 1
        assert state.data.shape[1] == GC_field.data(0).shape[1] + 2
        assert GC_field.data(0).shape[0] == GC_field.data(1).shape[0] + 1
        assert GC_field.data(0).shape[1] == GC_field.data(1).shape[1] - 1
        # TODO: assert G.data.shape == state.data.shape (but halo...)
        # TODO assert halo

        prev = scalar_field.clone(state)  # TODO rename?
        GC_antidiff = vector_field.clone(GC_field)
        flux = vector_field.clone(GC_field)
        halo = state.halo
        mpdata = MPDATA(curr=state, prev=prev, G=g_factor, GC_physical=GC_field, GC_antidiff=GC_antidiff, flux=flux,
                        n_iters=n_iters, halo=halo)

        return mpdata

    @staticmethod
    def kinematic_2d(grid, size, dt, stream_function: callable, field_values: dict, g_factor: np.ndarray, halo=1):
        GC = nondivergent_vector_field_2d(grid, size, halo, dt, stream_function)
        G = ScalarField(g_factor, halo=0)

        mpdatas = {}
        for key, value in field_values.items():
            state = uniform_scalar_field(grid, value, halo)
            mpdatas[key] = MPDATAFactory.mpdata(state=state, GC_field=GC, g_factor=G, n_iters=1)

        eulerian_fields = EulerianFields(mpdatas)
        return GC, eulerian_fields


def uniform_scalar_field(grid, value, halo):
    data = np.full(grid, value)
    scalar_field = ScalarField(data=data, halo=halo)
    return scalar_field


# TODO: move asserts to a unit test
def x_vec_coord(grid, size):
    nx = grid[0]+1
    nz = grid[1]
    xX = np.repeat(np.linspace(0, grid[0], nx).reshape((nx, 1)), nz, axis=1) / grid[0]
    assert np.amin(xX) == 0
    assert np.amax(xX) == 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(np.linspace(1 / 2, grid[1] - 1/2, nz).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) >= 0
    assert np.amax(zZ) <= 1
    assert zZ.shape == (nx, nz)
    return xX, zZ

# TODO: move asserts to a unit test
def z_vec_coord(grid, size):
    nx = grid[0]
    nz = grid[1]+1
    xX = np.repeat(np.linspace(1/2, grid[0]-1/2, nx).reshape((nx, 1)), nz, axis=1) / grid[0]
    assert np.amin(xX) >= 0
    assert np.amax(xX) <= 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(np.linspace(0, grid[1], nz).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) == 0
    assert np.amax(zZ) == 1
    assert zZ.shape == (nx, nz)
    return xX, zZ

def nondivergent_vector_field_2d(grid, size, halo, dt, stream_function: callable):
    # TODO: density!
    dx = size[0] / grid[0]
    dz = size[1] / grid[1]
    dxX = 1 / grid[0]
    dzZ = 1 / grid[1]

    xX, zZ = x_vec_coord(grid, size)
    rho_velocity_x = -(stream_function(xX, zZ + dzZ/2) - stream_function(xX, zZ - dzZ/2)) / dz

    xX, zZ = z_vec_coord(grid, size)
    rho_velocity_z = (stream_function(xX + dxX/2, zZ) - stream_function(xX - dxX/2, zZ)) / dx

    GC = [rho_velocity_x * dt / dx, rho_velocity_z * dt / dz]

    # CFL condition
    for d in range(len(GC)):
        np.testing.assert_array_less(np.abs(GC[d]), 1)

    result = VectorField(data=GC, halo=halo)

    # nondivergence (of velocity field, hence dt)
    assert np.amax(abs(vector_field.div(result, (dt, dt)).data)) < 5e-9

    return result
