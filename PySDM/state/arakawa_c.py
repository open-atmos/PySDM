import numpy as np


class Fields:
    def __init__(self, environment, stream_function):
        self.g_factor = make_rhod(environment.mesh.grid, environment.rhod_of)
        self.advector = nondivergent_vector_field_2d(
            environment.mesh.grid, environment.mesh.size, environment.dt, stream_function)
        self.advectees = dict(
            (key, np.full(environment.mesh.grid, value)) for key, value in environment.field_values.items())
        Z_COORD = 1
        self.courant_field = (
            self.advector[0] / environment.rhod_of(zZ=x_vec_coord(environment.mesh.grid)[Z_COORD]),
            self.advector[1] / environment.rhod_of(zZ=z_vec_coord(environment.mesh.grid)[Z_COORD])
        )


def nondivergent_vector_field_2d(grid, size, dt, stream_function: callable):
    dx = size[0] / grid[0]
    dz = size[1] / grid[1]
    dxX = 1 / grid[0]
    dzZ = 1 / grid[1]

    xX, zZ = x_vec_coord(grid)
    rho_velocity_x = -(stream_function(xX, zZ + dzZ/2) - stream_function(xX, zZ - dzZ/2)) / dz

    xX, zZ = z_vec_coord(grid)
    rho_velocity_z = (stream_function(xX + dxX/2, zZ) - stream_function(xX - dxX/2, zZ)) / dx

    rho_times_courant = [rho_velocity_x * dt / dx, rho_velocity_z * dt / dz]
    return rho_times_courant


def x_vec_coord(grid):
    nx = grid[0]+1
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


def z_vec_coord(grid):
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


def z_scalar_coord(grid):
    zZ = np.linspace(1/2, grid[-1]-1/2, grid[-1])
    return zZ


def make_rhod(grid, rhod_of_zZ):
    return np.repeat(
        rhod_of_zZ(
            z_scalar_coord(grid) / grid[-1]
        ).reshape((1, grid[1])),
        grid[0],
        axis=0
    )

