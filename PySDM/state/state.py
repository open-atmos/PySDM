"""
Created at 03.06.2019
"""

import numpy as np
from PySDM.attributes.attribute import Attribute


class State:

    def __init__(self, idx, n: (np.ndarray, Attribute), attributes, keys: dict, intensive_start: int,
                 cell_id: (np.ndarray, Attribute), cell_start: np.ndarray,
                 cell_origin: (np.ndarray, None), position_in_cell: (np.ndarray, None),
                 particles, whole_attributes=None):
        self.particles = particles
        self.__backend = particles.backend
        Storage = self.__backend.Storage

        self.__n_sd = particles.n_sd
        self.healthy = True
        self.__healthy_memory = Storage.from_ndarray(np.full((1,), 1))
        self.__idx = idx
        self.__strides = Storage.from_ndarray(self.particles.mesh.strides)

        # Dived into 2 arrays
        self.attributes = attributes
        self.keys = keys
        self.intensive_start = intensive_start

        self.__cell_start = Storage.from_ndarray(cell_start)
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
        self.sanitize()  # TODO: remove
        return len(self.__idx)

    def sanitize(self):
        if not self.healthy:
            self['n'].remove_zeros()
            self.healthy = True
            self.__healthy_memory[:] = 1
            self.__sorted = False

    def __getitem__(self, item):
        return self.whole_attributes[item].get()

    def permutation_global(self, u01):
        """
        apply Fisher-Yates algorithm to all super-droplets
        """
        self.__idx.shuffle(u01)
        self.__sorted = False

    def permutation_local(self, u01):
        """
        apply Fisher-Yates algorithm per cell
        """
        self.__idx.shuffle(u01, parts=self.cell_start)

    def __sort_by_cell_id(self):
        self.__cell_caretaker(self['cell id'], self.__cell_start, self.__idx, self.SD_num)
        self.__sorted = True

    def get_extensive_attrs(self):
        result = self.attributes[:self.intensive_start]
        return result

    def get_intensive_attrs(self):
        result = self.attributes[self.intensive_start:]
        return result

    def recalculate_cell_id(self):
        if 'cell origin' not in self.whole_attributes:
            return
        else:
            self.__backend.cell_id(self['cell id'], self['cell origin'], self.__strides)
            self.__sorted = False

    def moments(self, moment_0, moments, specs: dict, attr_name='volume', attr_range=(-np.inf, np.inf)):
        specs_idx, specs_rank = [], []
        for attr in specs:
            for rank in specs[attr]:
                specs_idx.append(self.keys[attr])
                specs_rank.append(rank)
        specs_idx = self.__backend.Storage.from_ndarray(np.array(specs_idx, dtype=int))
        specs_rank = self.__backend.Storage.from_ndarray(np.array(specs_rank, dtype=float))
        self.__backend.moments(moment_0, moments, self['n'], self.attributes, self['cell id'], self.__idx,
                               self.SD_num, specs_idx, specs_rank, attr_range[0], attr_range[1],
                               self.keys[attr_name])

    def find_pairs(self, is_first_in_pair):
        is_first_in_pair.find_pairs(self.cell_start, self['cell id'])

    def sum_pair(self, output, x: str, is_first_in_pair):
        output.sum_pair(self[x], is_first_in_pair)

    def max_pair(self, prob, is_first_in_pair):
        prob.max_pair(self['n'], is_first_in_pair)

    def coalescence(self, gamma, adaptive, subs, adaptive_memory):
        result = self.__backend.coalescence(n=self['n'],
                                            volume=self['volume'],
                                            idx=self.__idx,
                                            length=self.SD_num,
                                            intensive=self.get_intensive_attrs(),
                                            extensive=self.get_extensive_attrs(),
                                            gamma=gamma,
                                            healthy=self.__healthy_memory,
                                            adaptive=adaptive,
                                            subs=subs,
                                            adaptive_memory=adaptive_memory)
        self.healthy = bool(self.__healthy_memory)
        self.whole_attributes['volume'].mark_updated()
        return result

    def has_attribute(self, attr):
        return attr in self.keys

    def remove_precipitated(self):
        self.__backend.flag_precipitated(self['cell origin'], self['position in cell'],
                                         self.__idx, self.SD_num, self.__healthy_memory)
        self.healthy = bool(self.__healthy_memory)
