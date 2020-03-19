"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class State:

    def __init__(self, n: np.ndarray, attributes: dict, keys: dict, intensive_start: int,
                 cell_id: np.ndarray, cell_start: np.ndarray,
                 cell_origin: (np.ndarray, None), position_in_cell: (np.ndarray, None),
                 particles):
        self.particles = particles
        self.__backend = particles.backend

        self.__n_sd = particles.n_sd
        self.healthy = self.__backend.from_ndarray(np.full((1,), 1))
        self.__idx = self.__backend.from_ndarray(np.arange(self.SD_num))
        self.__tmp_idx = self.__backend.from_ndarray(np.arange(self.SD_num))
        self.n = self.__backend.from_ndarray(n)
        self.attributes = attributes
        self.keys = keys
        self.intensive_start = intensive_start
        self.position_in_cell = None if position_in_cell is None else self.__backend.from_ndarray(position_in_cell)
        self.cell_origin = None if cell_origin is None else self.__backend.from_ndarray(
            cell_origin)
        self.cell_id = self.__backend.from_ndarray(cell_id)
        self.__cell_start = self.__backend.from_ndarray(cell_start)
        # TODO!
        self.__cell_start_p = self.__backend.array((self.__backend.num_threads(), len(cell_start)), dtype=int)
        self.__sorted = False

    @property
    def cell_start(self):
        if not self.__sorted:
            self.__sort_by_cell_id()
        return self.__cell_start

    @property
    def SD_num(self):
        if not self.is_healthy():
            self.__n_sd = self.__backend.remove_zeros(self.n, self.__idx, length=self.__n_sd)
            self.healthy = self.__backend.from_ndarray(np.full((1,), 1))
            self.__sorted = False
        return self.__n_sd

    def get_backend_storage(self, item):
        attr = self.keys[item]
        result = self.__backend.read_row(self.attributes, attr)
        return result

    def permutation_global(self, u01):
        """
        apply Fisher-Yates algorithm to all super-droplets
        """
        self.__backend.shuffle_global(idx=self.__idx, length=self.SD_num, u01=u01)
        self.__sorted = False

    def permutation_local(self, u01):
        """
        apply Fisher-Yates algorithm per cell
        """
        self.__backend.shuffle_local(idx=self.__idx, u01=u01, cell_start=self.cell_start)

    def __sort_by_cell_id(self):
        self.__backend.counting_sort_by_cell_id(self.__tmp_idx, self.__idx, self.cell_id, self.SD_num, self.__cell_start)
        self.__idx, self.__tmp_idx = self.__tmp_idx, self.__idx
        self.__sorted = True

    def min(self, item):
        result = self.__backend.amin(self.get_backend_storage(item), self.__idx, self.SD_num)
        return result

    def max(self, item):
        result = self.__backend.amax(self.get_backend_storage(item), self.__idx, self.SD_num)
        return result

    def get_extensive_attrs(self):
        result = self.attributes[:self.intensive_start]
        return result

    def get_intensive_attrs(self):
        result = self.attributes[self.intensive_start:]
        return result

    def is_healthy(self):
        result = not self.__backend.first_element_is_zero(self.healthy)
        return result

    def recalculate_cell_id(self):
        if self.cell_origin is None:
            return
        else:
            self.__backend.cell_id(self.cell_id, self.cell_origin, self.particles.mesh.strides)
            self.__sorted = False

    def moments(self, moment_0, moments, specs: dict, attr_name='volume', attr_range=(-np.inf, np.inf)):
        specs_idx, specs_rank = [], []
        for attr in specs:
            for rank in specs[attr]:
                specs_idx.append(self.keys[attr])
                specs_rank.append(rank)
        specs_idx = np.array(specs_idx, dtype=int)
        specs_rank = np.array(specs_rank, dtype=float)
        self.__backend.moments(moment_0, moments, self.n, self.attributes, self.cell_id, self.__idx,
                               self.SD_num, specs_idx, specs_rank, attr_range[0], attr_range[1],
                               self.keys[attr_name])

    def find_pairs(self, cell_start, is_first_in_pair):
        self.__backend.find_pairs(cell_start, is_first_in_pair,
                                  self.cell_id,
                                  self.__idx,
                                  self.SD_num)

    def sum_pair(self, output, x, is_first_in_pair):
        self.__backend.sum_pair(output, self.get_backend_storage(x),
                                is_first_in_pair,
                                self.__idx,
                                self.SD_num)

    def max_pair(self, prob, is_first_in_pair):
        self.__backend.max_pair(prob, self.n, is_first_in_pair, self.__idx, self.SD_num)

    def coalescence(self, gamma):
        self.__backend.coalescence(n=self.n,
                                   volume=self.get_backend_storage('volume'),
                                   idx=self.__idx,
                                   length=self.SD_num,
                                   intensive=self.get_intensive_attrs(),
                                   extensive=self.get_extensive_attrs(),
                                   gamma=gamma,
                                   healthy=self.healthy)

    def has_attribute(self, attr):
        return attr in self.keys

    def remove_precipitated(self):
        self.__backend.flag_precipitated(self.cell_origin, self.position_in_cell, self.__idx, self.SD_num, self.healthy)
