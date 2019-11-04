"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.simulation.state import State
import numpy as np


class Advection:
    # TODO Adapter
    def __init__(self, n_sd, courant_field, backend, scheme='FTBS'):
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
        self.temp = backend.from_ndarray(np.zeros((n_sd, self.dimension), dtype=np.int64))

    def __call__(self, state: State):
        # TODO
        cell_origin = state.cell_origin[state.idx[:state.SD_num]]
        position_in_cell = state.position_in_cell[state.idx[:state.SD_num]]
        cell_id = state.cell_id[state.idx[:state.SD_num]]
        displacement = self.displacement[:state.SD_num]
        # displacement = self.displacement
        # cell_origin = state.cell_origin
        # position_in_cell = state.position_in_cell
        # cell_id = state.cell_id

        self.calculate_displacement(displacement, self.courant, cell_origin, position_in_cell)
        self.update_position(position_in_cell, displacement)
        self.update_cell_origin(cell_origin, position_in_cell)
        self.boundary_condition(cell_origin)
        self.recalculate_cell_id(cell_id, cell_origin, state.grid)

        for i in range(state.SD_num):
            xdi = state.idx[i]
            state.cell_origin[xdi] = cell_origin[i]
            state.position_in_cell[xdi] = position_in_cell[i]
            state.cell_id[xdi] = cell_id[i]

    def calculate_displacement(self, displacement, courant, cell_origin, position_in_cell):
        # TODO: move to backend
        SD_num = self.backend.shape(displacement)[0]
        for droplet in range(SD_num):
            for d in range(self.dimension):
                C_l = courant[d][
                    cell_origin[droplet, 0],
                    cell_origin[droplet, 1]
                ]
                C_r = courant[d][
                    cell_origin[droplet, 0] + 1 * (d == 0),
                    cell_origin[droplet, 1] + 1 * (d == 1)
                ]
                omega = position_in_cell[droplet, d]
                if self.scheme == 'FTFS':
                    displacement[droplet, d] = (
                        C_l * (1 - omega) +
                        C_r * omega
                    )
                elif self.scheme == 'FTBS':
                    # see eqs 14-16 in Arabas et al. 2015 (libcloudph++)
                    dC = C_r - C_l
                    displacement[droplet, d] = (omega * dC + C_l) / (1 - dC)
                else:
                    raise NotImplementedError()

    def update_position(self, position_in_cell, displacement):
        self.backend.sum(position_in_cell, displacement)

    def update_cell_origin(self, cell_origin, position_in_cell):
        # TODO add backend.add_floor/subtract_floor ?
        floor_of_position = self.temp[:position_in_cell.shape[0]]

        self.backend.floor2(floor_of_position, position_in_cell)
        self.backend.sum(cell_origin, floor_of_position)
        self.backend.multiply(floor_of_position, -1)
        self.backend.sum(position_in_cell, floor_of_position)

    def boundary_condition(self, cell_origin):
        # TODO: hardcoded periodic
        self.backend.modulo(cell_origin, self.grid)

    def recalculate_cell_id(self, cell_id, cell_origin, grid):
        self.backend.cell_id(cell_id, cell_origin, grid)
