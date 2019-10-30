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
    def __init__(self, n_sd, courant_field, backend, scheme='BackwardEuler'):
        if len(courant_field) == 2:
            assert courant_field[0].shape[0] == courant_field[1].shape[0] + 1
            assert courant_field[0].shape[1] == courant_field[1].shape[1] - 1
        else:
            raise NotImplementedError()

        # CFL
        for d in range(len(courant_field)):
            assert np.amax(abs(courant_field[d])) <= 1

        self.backend = backend
        self.scheme = scheme
        self.grid = np.array([courant_field[1].shape[0], courant_field[0].shape[1]])
        self.dimension = len(courant_field)
        self.courant = [backend.from_ndarray(courant_field[i]) for i in range(self.dimension)]

        self.displacement = backend.from_ndarray(np.zeros((n_sd, self.dimension)))
        self.floor_of_positions = backend.from_ndarray(np.zeros((n_sd, self.dimension), dtype=int))

    def __call__(self, state: State):
        # TODO
        cell_origin = state.cell_origin[state.idx[:state.SD_num]]
        position_in_cell = state.position_in_cell[state.idx[:state.SD_num]]
        cell_id = state.cell_id[state.idx[:state.SD_num]]

        self.calculate_displacement(cell_origin, position_in_cell)
        self.update_position(position_in_cell)
        self.update_cell_origin(cell_origin, position_in_cell)
        self.boundary_condition(cell_origin)
        self.recalculate_cell_id(cell_id, cell_origin, state.grid)

        state.cell_origin[state.idx[:state.SD_num]] = cell_origin
        state.position_in_cell[state.idx[:state.SD_num]] = position_in_cell
        state.cell_id[state.idx[:state.SD_num]] = cell_id

    def calculate_displacement(self, cell_origin, position_in_cell):
        # TODO: move to backend
        SD_num = self.backend.shape(cell_origin)[0]
        for droplet in range(SD_num):
            for d in range(self.dimension):
                C_l = self.courant[d][
                    cell_origin[droplet, 0],
                    cell_origin[droplet, 1]
                ]
                C_r = self.courant[d][
                    cell_origin[droplet, 0] + 1 * (d == 0),
                    cell_origin[droplet, 1] + 1 * (d == 1)
                ]
                omega = position_in_cell[droplet, d]
                if self.scheme == 'ForwardEuler':
                    self.displacement[droplet, d] = (
                        C_l * (1 - omega) +
                        C_r * omega
                    )
                elif self.scheme == 'BackwardEuler':
                    # see eqs 14-16 in Arabas et al. 2015 (libcloudph++)
                    dC = C_r - C_l
                    self.displacement[droplet, d] = (omega * dC + C_l) / (1 - dC)
                else:
                    raise NotImplementedError()

    def update_position(self, position_in_cell):
        self.backend.sum(position_in_cell, self.displacement)

    def update_cell_origin(self, cell_origin, position_in_cell):
        # TODO add backend.add_floor/subtract_floor ?
        self.backend.floor2(self.floor_of_positions, position_in_cell)
        self.backend.sum(cell_origin, self.floor_of_positions)
        self.backend.multiply(self.floor_of_positions, -1)
        self.backend.sum(position_in_cell, self.floor_of_positions)

    def boundary_condition(self, cell_origin):
        # TODO: hardcoded periodic
        self.backend.modulo(cell_origin, self.grid)

    def recalculate_cell_id(self, cell_id, cell_origin, grid):
        self.backend.cell_id(cell_id, cell_origin, grid)
