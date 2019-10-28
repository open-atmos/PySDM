"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from SDM.simulation.state import State
import numpy as np


class ExplicitEulerWithInterpolation:
    # TODO Adapter
    def __init__(self, n_sd, courant_field, backend):
        if len(courant_field) == 2:
            assert courant_field[0].shape[0] == courant_field[1].shape[0] + 1
            assert courant_field[0].shape[1] == courant_field[1].shape[1] - 1
        else:
            raise NotImplementedError()

        # CFL
        for d in range(len(courant_field)):
            assert np.amax(abs(courant_field[d])) <= 1

        self.backend = backend

        self.grid = np.array([courant_field[1].shape[0], courant_field[0].shape[1]])
        self.dimension = len(courant_field)
        self.courant = [backend.from_ndarray(courant_field[i]) for i in range(self.dimension)]

        self.displacement = backend.from_ndarray(np.zeros((n_sd, self.dimension)))
        self.floor_of_positions = backend.from_ndarray(np.zeros((n_sd, self.dimension), dtype=int))

    def __call__(self, state: State):
        self.interpolation(state)
        self.update_position(state)
        self.update_cell_origin(state)
        self.boundary_condition(state)
        self.recalculate_cell_id(state)

    def interpolation(self, state: State):
        # TODO: idx-aware indexing
        # TODO: move to backend
        for droplet in range(state.SD_num):
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

    def update_position(self, state: State):
        self.backend.sum(state.position_in_cell, self.displacement)

    def update_cell_origin(self, state: State):
        # TODO add backend.add_floor/subtract_floor ?
        self.backend.floor2(self.floor_of_positions, state.position_in_cell)
        self.backend.sum(state.cell_origin, self.floor_of_positions)
        self.backend.multiply(self.floor_of_positions, -1)
        self.backend.sum(state.position_in_cell, self.floor_of_positions)

    def boundary_condition(self, state: State):
        # TODO: hardcoded periodic
        self.backend.modulo(state.cell_origin, self.grid)

    def recalculate_cell_id(self, state: State):
        self.backend.cell_id(state.cell_id, state.cell_origin, state.grid)
