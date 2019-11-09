"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.simulation.state import State


class TestableState(State):
    @staticmethod
    def state_0d(n: np.ndarray, intensive: dict, extensive: dict, backend):
        cell_id = np.zeros_like(n)
        state = TestableState(n, (), intensive, extensive, cell_id, None, None, backend)
        return state

    @staticmethod
    def state_2d(n: np.ndarray, grid: tuple, intensive: dict, extensive: dict, positions: np.ndarray, backend):
        cell_origin = positions.astype(dtype=int)
        position_in_cell = positions - np.floor(positions)
        cell_id = np.empty_like(n)
        state = TestableState(n, grid, intensive, extensive, cell_id, cell_origin, position_in_cell, backend)
        state.recalculate_cell_id()
        return state

    def __getitem__(self, item: str):
        idx = self.backend.to_ndarray(self.idx)
        all_valid = idx[:self.SD_num]
        if item == 'n':
            n = self.backend.to_ndarray(self.n)
            result = n[all_valid]
        elif item == 'cell_id':
            cell_id = self.backend.from_ndarray(self.cell_id)
            result = cell_id[all_valid]
        else:
            tensive = self.keys[item][0]
            attr = self.keys[item][1]
            attribute = self.backend.to_ndarray(self.backend.read_row(self.attributes[tensive], attr))
            result = attribute[all_valid]
        return result
