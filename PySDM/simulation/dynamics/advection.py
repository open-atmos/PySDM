"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State
import numpy as np


class Advection:
    # TODO Adapter
    def __init__(self, simulation, courant_field, scheme='FTBS'):
        if len(courant_field) == 2:
            assert courant_field[0].shape[0] == courant_field[1].shape[0] + 1
            assert courant_field[0].shape[1] == courant_field[1].shape[1] - 1
        else:
            raise NotImplementedError()

        # CFL
        for d in range(len(courant_field)):
            assert np.amax(abs(courant_field[d])) <= 1

        if scheme == 'FTFS':
            method = simulation.backend.explicit_in_space
        elif scheme == 'FTBS':
            method = simulation.backend.implicit_in_space
        else:
            raise NotImplementedError()

        self.simulation = simulation
        self.scheme = method

        self.dimension = len(courant_field)
        self.grid = np.array([courant_field[1].shape[0], courant_field[0].shape[1]])

        self.courant = [self.simulation.backend.from_ndarray(courant_field[i]) for i in range(self.dimension)]

        self.displacement = self.simulation.backend.from_ndarray(np.zeros((self.simulation.n_sd, self.dimension)))
        self.temp = self.simulation.backend.from_ndarray(np.zeros((self.simulation.n_sd, self.dimension), dtype=np.int64))

    def __call__(self):
        # TODO: not need all array only [idx[:sd_num]]
        displacement = self.displacement
        cell_origin = self.simulation.state.cell_origin
        position_in_cell = self.simulation.state.position_in_cell

        self.calculate_displacement(displacement, self.courant, cell_origin, position_in_cell)
        self.update_position(position_in_cell, displacement)
        self.update_cell_origin(cell_origin, position_in_cell)
        self.boundary_condition(cell_origin)
        self.simulation.state.recalculate_cell_id()

    def calculate_displacement(self, displacement, courant, cell_origin, position_in_cell):
        for dim in range(self.dimension):
            self.simulation.backend.calculate_displacement(dim, self.scheme, displacement, courant[dim], cell_origin, position_in_cell)

    def update_position(self, position_in_cell, displacement):
        self.simulation.backend.add(position_in_cell, displacement)

    def update_cell_origin(self, cell_origin, position_in_cell):
        # TODO add backend.add_floor/subtract_floor ?
        floor_of_position = self.temp[:position_in_cell.shape[0]]

        self.simulation.backend.floor2(floor_of_position, position_in_cell)
        self.simulation.backend.add(cell_origin, floor_of_position)
        self.simulation.backend.multiply(floor_of_position, -1)
        self.simulation.backend.add(position_in_cell, floor_of_position)

    def boundary_condition(self, cell_origin):
        # TODO: hardcoded periodic
        self.simulation.backend.column_modulo(cell_origin, self.grid)
