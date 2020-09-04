"""
Created at 04.09.2020
"""
import numpy as np
from MPyDATA.arakawa_c.discretisation import x_vec_coord, z_vec_coord


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


def make_rhod(grid, rhod_of):
    return np.repeat(
        rhod_of(
            (np.arange(grid[1]) + 1 / 2) / grid[1]
        ).reshape((1, grid[1])),
        grid[0],
        axis=0
    )