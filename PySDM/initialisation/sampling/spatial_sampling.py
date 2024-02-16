"""
spatial sampling logic (i.e., physical x-y-z coordinates)
"""

# TODO #305 QUASIRANDOM & GRID
#  http://www.ii.uj.edu.pl/~arabas/workshop_2019/files/talk_Shima.pdf


class Pseudorandom:  # pylint: disable=too-few-public-methods
    @staticmethod
    def sample(*, backend, grid, n_sd, z_part=None):
        dimension = len(grid)
        n_elements = dimension * n_sd

        storage = backend.Storage.empty(n_elements, dtype=float)
        backend.Random(seed=backend.formulae.seed, size=n_elements)(storage)
        positions = storage.to_ndarray().reshape(dimension, n_sd)

        if z_part is None:
            for dim in range(dimension):
                positions[dim, :] *= grid[dim]
        else:
            assert dimension == 1
            iz_min = int(grid[0] * z_part[0])
            iz_max = int(grid[0] * z_part[1])
            for dim in range(dimension):
                positions[dim, :] *= iz_max - iz_min
                positions[dim, :] += iz_min

        return positions
