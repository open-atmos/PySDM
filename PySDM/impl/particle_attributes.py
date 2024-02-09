"""
logic for handling particle attributes within
 `PySDM.particulator.Particulator`
"""

from typing import Dict

import numpy as np

from PySDM.attributes.impl.attribute import Attribute


class ParticleAttributes:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        particulator,
        idx,
        extensive_attribute_storage,
        extensive_keys: dict,
        cell_start,
        attributes: Dict[str, Attribute],
    ):
        self.__valid_n_sd = particulator.n_sd
        self.healthy = True
        self.__healthy_memory = particulator.Storage.from_ndarray(np.full((1,), 1))
        self.__idx = idx

        self.__extensive_attribute_storage = extensive_attribute_storage
        self.__extensive_keys = extensive_keys

        self.cell_idx = particulator.Index.identity_index(len(cell_start) - 1)
        self.__cell_start = particulator.Storage.from_ndarray(cell_start)
        self.__cell_caretaker = particulator.backend.make_cell_caretaker(
            self.__idx.shape,
            self.__idx.dtype,
            len(self.__cell_start),
            scheme=particulator.sorting_scheme,
        )
        self.__sorted = False
        self.__attributes = attributes

    @property
    def cell_start(self):
        if not self.__sorted:
            self.__sort_by_cell_id()
        return self.__cell_start

    @property
    def super_droplet_count(self):
        """returns the number of super-droplets in the system
        (which might differ from the initial one due to precipitation
        or removal during collision of multiplicity-of-one particles)"""
        assert self.healthy
        return len(self.__idx)

    def mark_updated(self, key):
        self.__attributes[key].mark_updated()

    def sanitize(self):
        if not self.healthy:
            self.__idx.length = self.__valid_n_sd
            self.__idx.remove_zero_n_or_flagged(self["multiplicity"])
            self.__valid_n_sd = self.__idx.length
            self.healthy = True
            self.__healthy_memory[:] = 1
            self.__sorted = False

    def cut_working_length(self, length):
        assert length <= len(self.__idx)
        self.__idx.length = length

    def get_working_length(self):
        return len(self.__idx)

    def reset_working_length(self):
        self.__idx.length = self.__valid_n_sd

    def reset_cell_idx(self):
        self.cell_idx.reset_index()
        self.__sort_by_cell_id()

    def keys(self):
        return self.__attributes.keys()

    def __getitem__(self, item):
        return self.__attributes[item].get()

    def __contains__(self, key):
        return key in self.__attributes

    def permutation(self, u01, local):
        """apply Fisher-Yates algorithm to all super-droplets (local=False) or
        otherwise on a per-cell basis"""
        if local:
            self.__idx.shuffle(u01, parts=self.cell_start)
        else:
            self.__idx.shuffle(u01)
            self.__sorted = False

    def __sort_by_cell_id(self):
        self.__cell_caretaker(
            self["cell id"], self.cell_idx, self.__cell_start, self.__idx
        )
        self.__sorted = True

    def get_extensive_attribute_storage(self):
        return self.__extensive_attribute_storage

    def get_extensive_attribute_keys(self):
        return self.__extensive_keys.keys()

    def has_attribute(self, attr):
        return attr in self.__attributes
