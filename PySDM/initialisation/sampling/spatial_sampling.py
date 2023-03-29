"""
spatial sampling logic (i.e., physical x-y-z coordinates)
"""
# TODO #305 QUASIRANDOM & GRID
#  http://www.ii.uj.edu.pl/~arabas/workshop_2019/files/talk_Shima.pdf


class Pseudorandom:  # pylint: disable=too-few-public-methods
    @staticmethod
    def sample(*, backend, grid, n_sd):
        dimension = len(grid)
        n_elements = dimension * n_sd

        storage = backend.Storage.empty(n_elements, dtype=float)
        backend.Random(seed=backend.formulae.seed, size=n_elements)(storage)
        positions = storage.to_ndarray().reshape(dimension, n_sd)

        for dim in range(dimension):
            positions[dim, :] *= grid[dim]
        return positions
