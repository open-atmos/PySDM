"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class State:

    def __init__(self, n: np.ndarray, attributes: dict, keys: dict,
                 cell_id: np.ndarray, cell_origin: (np.ndarray, None), position_in_cell: (np.ndarray, None), particles):
        self.particles = particles
        self.__backend = particles.backend

        self.SD_num = particles.n_sd
        # TODO make private
        self.idx = self.__backend.from_ndarray(np.arange(self.SD_num))
        self.__tmp_idx = self.__backend.from_ndarray(np.arange(self.SD_num))
        self.n = self.__backend.from_ndarray(n)
        # TODO: 0=tensive, 1=index (also in moments)
        self.attributes = attributes
        self.keys = keys
        self.position_in_cell = None if position_in_cell is None else self.__backend.from_ndarray(position_in_cell)
        self.cell_origin = None if cell_origin is None else self.__backend.from_ndarray(cell_origin)  # TODO: move to advection? (or remove - same info in cell_id)
        self.cell_id = self.__backend.from_ndarray(cell_id)
        self.healthy = self.__backend.from_ndarray(np.full((1,), 1))

    def get_backend_storage(self, item):
        tensive = self.keys[item][0]
        attr = self.keys[item][1]
        result = self.__backend.read_row(self.attributes[tensive], attr)
        return result

    def unsort(self):
        # TODO: consider having two idx arrays and unsorting them asynchronously
        self.__backend.shuffle(idx=self.idx, length=self.SD_num, axis=0)

    def sort_by_cell_id(self, cell_start):
        self.__backend.countsort_by_cell_id(self.__tmp_idx, self.idx, self.cell_id, self.SD_num, cell_start)
        self.idx, self.__tmp_idx = self.__tmp_idx, self.idx

    def min(self, item):
        result = self.__backend.amin(self.get_backend_storage(item), self.idx, self.SD_num)
        return result

    def max(self, item):
        result = self.__backend.amax(self.get_backend_storage(item), self.idx, self.SD_num)
        return result

    def get_extensive_attrs(self):
        result = self.attributes['extensive']
        return result

    def get_intensive_attrs(self):
        result = self.attributes['intensive']
        return result

    def is_healthy(self):
        result = not self.__backend.first_element_is_zero(self.healthy)
        return result

    # TODO: optionally recycle n=0 drops
    def housekeeping(self):
        if not self.is_healthy():
            self.SD_num = self.__backend.remove_zeros(self.n, self.idx, length=self.SD_num)
            self.healthy = self.__backend.from_ndarray(np.full((1,), 1))

    def recalculate_cell_id(self):
        if self.cell_origin is None:
            return
        else:
            self.__backend.cell_id(self.cell_id, self.cell_origin, self.particles.mesh.strides)

    def moments(self, moment_0, moments, specs: dict, attr_name='volume', attr_range=(0, np.inf)):
        # TODO: intensive
        specs_idx, specs_rank = [], []
        for attr in specs:
            for rank in specs[attr]:
                specs_idx.append(self.keys[attr][1])
                specs_rank.append(rank)
        specs_idx = np.array(specs_idx, dtype=int)
        specs_rank = np.array(specs_rank, dtype=float)
        self.__backend.moments(moment_0, moments, self.n, self.get_extensive_attrs(), self.cell_id, self.idx,
                               self.SD_num, specs_idx, specs_rank, attr_range[0], attr_range[1],
                               self.keys[attr_name][1])
