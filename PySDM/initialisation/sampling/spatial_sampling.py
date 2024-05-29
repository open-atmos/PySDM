"""
spatial sampling logic (i.e., physical x-y-z coordinates)
"""

# TODO #305 QUASIRANDOM & GRID
#  http://www.ii.uj.edu.pl/~arabas/workshop_2019/files/talk_Shima.pdf


class Pseudorandom:  # pylint: disable=too-few-public-methods
    @staticmethod
    def sample(*, backend, grid, n_sd, z_part=None, x_part=None):
        n_dims = len(grid)
        scale_fac = (
            []
        )  # scale factor for each dimension; y = scale_factor * eps + affine_factor
        affine_fac = []  # affine factor for each dimension

        storage = backend.Storage.empty(n_dims * n_sd, dtype=float)
        backend.Random(seed=backend.formulae.seed, size=n_dims * n_sd)(storage)
        positions = storage.to_ndarray().reshape(n_dims, n_sd)

        if z_part is None:
            scale_fac.append(grid[0])
            affine_fac.append(0)
        else:
            iz_min = int(grid[0] * z_part[0])
            iz_max = int(grid[0] * z_part[1])
            scale_fac.append(iz_max - iz_min)
            affine_fac.append(iz_min)

        if x_part is not None:
            ix_min = int(grid[1] * x_part[0])
            ix_max = int(grid[1] * x_part[1])
            scale_fac.append(ix_max - ix_min)
            affine_fac.append(ix_min)
        else:
            if n_dims == 2:
                scale_fac.append(grid[1])
                affine_fac.append(0)

        for dim in range(n_dims):
            positions[dim, :] *= scale_fac[dim]
            positions[dim, :] += affine_fac[dim]

        return positions
