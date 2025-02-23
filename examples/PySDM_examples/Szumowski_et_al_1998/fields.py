import numpy as np

from PySDM.impl.arakawa_c import z_scalar_coord


def z_vec_coord(grid):
    nx = grid[0]
    nz = grid[1] + 1
    xX = (
        np.repeat(np.linspace(1 / 2, grid[0] - 1 / 2, nx).reshape((nx, 1)), nz, axis=1)
        / grid[0]
    )
    assert np.amin(xX) >= 0
    assert np.amax(xX) <= 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(np.linspace(0, grid[1], nz).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) == 0
    assert np.amax(zZ) == 1
    assert zZ.shape == (nx, nz)
    return xX, zZ


def x_vec_coord(grid):
    nx = grid[0] + 1
    nz = grid[1]
    xX = np.repeat(np.linspace(0, grid[0], nx).reshape((nx, 1)), nz, axis=1) / grid[0]
    assert np.amin(xX) == 0
    assert np.amax(xX) == 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(z_scalar_coord(grid).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) >= 0
    assert np.amax(zZ) <= 1
    assert zZ.shape == (nx, nz)
    return xX, zZ


def nondivergent_vector_field_2d(
    grid: tuple, size: tuple, dt: float, stream_function: callable, t
):
    dx = size[0] / grid[0]
    dz = size[1] / grid[1]
    dxX = 1 / grid[0]
    dzZ = 1 / grid[1]

    xX, zZ = x_vec_coord(grid)
    rho_velocity_x = (
        -(stream_function(xX, zZ + dzZ / 2, t) - stream_function(xX, zZ - dzZ / 2, t))
        / dz
    )

    xX, zZ = z_vec_coord(grid)
    rho_velocity_z = (
        stream_function(xX + dxX / 2, zZ, t) - stream_function(xX - dxX / 2, zZ, t)
    ) / dx

    rho_times_courant = [rho_velocity_x * dt / dx, rho_velocity_z * dt / dz]
    return rho_times_courant
