"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from SDM.simulation.state import State
import numpy as np


class Advection:
    # TODO Adapter
    def __init__(self, n_sd, courant_field: tuple, backend):
        if len(courant_field) == 2:
            assert courant_field[0].shape[0] == courant_field[1].shape[0] + 1
            assert courant_field[0].shape[1] == courant_field[1].shape[1] - 1
        else:
            raise NotImplementedError()

        self.backend = backend

        self.grid = np.array([courant_field[1].shape[0], courant_field[0].shape[1]])
        self.n_sd = n_sd
        self.dimension = len(courant_field)
        self.courant = [backend.from_ndarray(courant_field[i]) for i in range(self.dimension)]

        self.displacement = backend.from_ndarray(np.zeros((self.n_sd, self.dimension)))
        self.floor_of_positions = backend.from_ndarray(np.zeros((self.n_sd, self.dimension), dtype=int))

    def __call__(self, state: State):
        # interpolation
        for droplet in range(self.n_sd):
            for d in range(self.dimension):
                self.displacement[droplet, d] = (
                        self.courant[d][
                        state.cell_origin[droplet, 0],
                        state.cell_origin[droplet, 1]
                        ] * (0 + state.position_in_cell[droplet, d]) +
                        self.courant[d][
                        state.cell_origin[droplet, 0] + 1 * (d == 0),
                        state.cell_origin[droplet, 1] + 1 * (d == 1),
                        ] * (1 - state.position_in_cell[droplet, d])
                )
        # update position
        self.backend.sum(state.position_in_cell, self.displacement)

        # update cell_origin
        self.segments(state)

        # TODO handle boundary condition ( & invalid cell_origin?)

    def segments(self, state: State):
        # TODO add backend.add_floor/subtract_floor ?
        self.backend.floor2(self.floor_of_positions, state.position_in_cell)
        self.backend.sum(state.cell_origin, self.floor_of_positions)
        self.backend.multiply(self.floor_of_positions, -1)
        self.backend.sum(state.position_in_cell, self.floor_of_positions)

        self.backend.cell_id(state.cell_id, state.cell_origin, state.grid)

