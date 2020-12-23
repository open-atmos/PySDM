"""
Created at 03.06.2019
"""

import numpy as np


class Particles:

    def __init__(
            self, core,
            idx,
            base_attributes,
            keys: dict,
            intensive_start: int,
            cell_start,
            attributes: dict
    ):
        self.core = core

        self.__n_sd = core.n_sd
        self.healthy = True
        self.__healthy_memory = self.core.Storage.from_ndarray(np.full((1,), 1))
        self.__idx = idx
        self.__strides = self.core.Storage.from_ndarray(self.core.mesh.strides)

        # Dived into 2 arrays
        self.base_attributes = base_attributes
        self.keys = keys
        self.intensive_start = intensive_start

        self.__cell_start = self.core.Storage.from_ndarray(cell_start)
        self.__cell_caretaker = self.core.bck.make_cell_caretaker(self.__idx, self.__cell_start,
                                                                  scheme=core.sorting_scheme)
        self.__sorted = False
        self.attributes = attributes

        self.recalculate_cell_id()

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
        return self.attributes[item].get()

    def permutation(self, u01, local=True):
        if local:
            """
            apply Fisher-Yates algorithm per cell
            """
            self.__idx.shuffle(u01, parts=self.cell_start)
        else:
            """
            apply Fisher-Yates algorithm to all super-droplets
            """
            self.__idx.shuffle(u01)
            self.__sorted = False

    def __sort_by_cell_id(self):
        self.__cell_caretaker(self['cell id'], self.__cell_start, self.__idx, self.SD_num)
        self.__sorted = True

    def get_extensive_attrs(self):
        result = self.base_attributes[:self.intensive_start]
        return result

    def get_intensive_attrs(self):
        result = self.base_attributes[self.intensive_start:]
        return result

    def recalculate_cell_id(self):
        if 'cell origin' not in self.attributes:
            return
        else:
            self.core.bck.cell_id(self['cell id'], self['cell origin'], self.__strides)
            self.__sorted = False

    def moments(self, moment_0, moments, specs: dict, attr_name='volume', attr_range=(-np.inf, np.inf)):
        specs_idx, specs_rank = [], []
        for attr in specs:
            for rank in specs[attr]:
                specs_idx.append(self.keys[attr])
                specs_rank.append(rank)
        specs_idx = self.core.bck.Storage.from_ndarray(np.array(specs_idx, dtype=int))
        specs_rank = self.core.bck.Storage.from_ndarray(np.array(specs_rank, dtype=float))
        self.core.bck.moments(moment_0, moments, self['n'], self.base_attributes, self['cell id'], self.__idx,
                              self.SD_num, specs_idx, specs_rank, attr_range[0], attr_range[1],
                              self.keys[attr_name])

    def coalescence(self, gamma, adaptive, subs, adaptive_memory):
        result = self.core.bck.coalescence(n=self['n'],
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
        self.attributes['volume'].mark_updated()
        return result

    def has_attribute(self, attr):
        return attr in self.keys

    def remove_precipitated(self) -> float:
        res = self.core.bck.flag_precipitated(self['cell origin'], self['position in cell'],
                                              self['volume'], self['n'],
                                              self.__idx, self.SD_num, self.__healthy_memory)
        self.healthy = bool(self.__healthy_memory)
        return res
