"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.attributes.attribute import Attribute


class State:

    def __init__(self, n: (np.ndarray, Attribute), attributes: dict, keys: dict, intensive_start: int,
                 cell_id: (np.ndarray, Attribute), cell_start: np.ndarray,
                 cell_origin: (np.ndarray, None), position_in_cell: (np.ndarray, None),
                 particles, whole_attributes=None):
        self.particles = particles
        self.__backend = particles.backend

        self.__n_sd = particles.n_sd
        self.healthy = True
        self.__healthy_memory = self.__backend.from_ndarray(np.full((1,), 1))
        self.__idx = self.__backend.from_ndarray(np.arange(self.SD_num))

        # Dived into 2 arrays
        self.attributes = attributes
        self.keys = keys
        self.intensive_start = intensive_start

        self.__cell_start = self.__backend.from_ndarray(cell_start)
        self.__cell_caretaker = self.__backend.make_cell_caretaker(self.__idx, self.__cell_start,
                                                                   scheme=particles.sorting_scheme)
        self.__sorted = False

        self.whole_attributes = whole_attributes

    @property
    def cell_start(self):
        if not self.__sorted:
            self.__sort_by_cell_id()
        return self.__cell_start

    @property
    def SD_num(self):
        if not self.healthy:
            print("preremove")
            self.__n_sd = self.__backend.remove_zeros(self['n'], self.__idx, length=self.__n_sd)
            print("remove")
            self.healthy = True
            self.__healthy_memory = self.__backend.from_ndarray(np.full((1,), 1))
            self.__sorted = False
        return self.__n_sd

    def __getitem__(self, item):
        return self.whole_attributes[item].get()

    def permutation_global(self, u01):
        """
        apply Fisher-Yates algorithm to all super-droplets
        """
        # self.__backend.shuffle_global(idx=self.__idx, length=self.SD_num, u01=u01)
        # self.__sorted = False

    def permutation_local(self, u01):
        """
        apply Fisher-Yates algorithm per cell
        """
        # self.__backend.shuffle_local(idx=self.__idx, u01=u01, cell_start=self.cell_start)

    def __sort_by_cell_id(self):
        self.__idx = self.__cell_caretaker(self['cell id'], self.__cell_start, self.__idx, self.SD_num)
        self.__sorted = True

    def min(self, item):
        result = self.__backend.amin(self[item], self.__idx, self.SD_num)
        return result

    def max(self, item):
        result = self.__backend.amax(self[item], self.__idx, self.SD_num)
        return result

    def get_extensive_attrs(self):
        result = self.__backend.range(self.attributes, stop=self.intensive_start)
        return result

    def get_intensive_attrs(self):
        result = self.__backend.range(self.attributes, start=self.intensive_start)
        return result

    def recalculate_cell_id(self):
        if 'cell origin' not in self.whole_attributes:
            return
        else:
            self.__backend.cell_id(self['cell id'], self['cell origin'], self.particles.mesh.strides)
            self.__sorted = False

    def moments(self, moment_0, moments, specs: dict, attr_name='volume', attr_range=(-np.inf, np.inf)):
        specs_idx, specs_rank = [], []
        for attr in specs:
            for rank in specs[attr]:
                specs_idx.append(self.keys[attr])
                specs_rank.append(rank)
        specs_idx = self.__backend.from_ndarray(np.array(specs_idx, dtype=int))
        specs_rank = self.__backend.from_ndarray(np.array(specs_rank, dtype=float))
        self.__backend.moments(moment_0, moments, self['n'], self.attributes, self['cell id'], self.__idx,
                               self.SD_num, specs_idx, specs_rank, attr_range[0], attr_range[1],
                               self.keys[attr_name])

    def find_pairs(self, cell_start, is_first_in_pair):
        self.__backend.find_pairs(cell_start, is_first_in_pair,
                                  self['cell id'],
                                  self.__idx,
                                  self.SD_num)

    def sum_pair(self, output, x: str, is_first_in_pair):
        self.__backend.sum_pair(output, self[x],
                                is_first_in_pair,
                                self.__idx,
                                self.SD_num)

    def max_pair(self, prob, is_first_in_pair):
        self.__backend.max_pair(prob, self['n'], is_first_in_pair, self.__idx, self.SD_num)

    def coalescence(self, gamma):
        self.__backend.coalescence(n=self['n'],
                                   volume=self['volume'],
                                   idx=self.__idx,
                                   length=self.SD_num,
                                   intensive=self.get_intensive_attrs(),
                                   extensive=self.get_extensive_attrs(),
                                   gamma=gamma,
                                   healthy=self.__healthy_memory)
        self.healthy = not self.__backend.first_element_is_zero(self.__healthy_memory)
        self.whole_attributes['volume'].mark_updated()

    def has_attribute(self, attr):
        return attr in self.keys

    def remove_precipitated(self):
        self.__backend.flag_precipitated(self['cell origin'], self['position in cell'],
                                         self.__idx, self.SD_num, self.__healthy_memory)
        self.healthy = not self.__backend.first_element_is_zero(self.__healthy_memory)
