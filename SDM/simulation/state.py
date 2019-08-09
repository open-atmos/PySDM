"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from SDM.backends.default import Default


class State:
    def __init__(self, n: np.ndarray, intensive: dict, extensive: dict, segment_num: int, backend):
        assert State.check_args(n, intensive, extensive)

        self.backend = backend

        self.SD_num = len(n)
        self.idx = backend.from_ndarray(np.arange(self.SD_num))
        self.n = backend.from_ndarray(n)
        self.attributes, self.keys = State.init_attributes_and_keys(self.backend, intensive, extensive, self.SD_num)
        self.healthy = backend.from_ndarray(np.full((1,), 1))

    @staticmethod
    def check_args(n, intensive, extensive):
        result = True
        if n.ndim != 1:
            result = False
        # https://en.wikipedia.org/wiki/Intensive_and_extensive_properties
        for attribute in (*intensive.values(), *extensive.values()):
            if attribute.shape != n.shape:
                result = False
        return result

    @staticmethod
    def init_attributes_and_keys(backend, intensive: dict, extensive: dict, SD_num) -> (dict, dict):
        attributes = {'intensive': backend.array((len(intensive), SD_num), float),
                      'extensive': backend.array((len(extensive), SD_num), float)
                      }
        keys = {}

        for tensive in attributes:
            idx = 0
            for key, array in {'intensive': intensive, 'extensive': extensive}[tensive].items():
                keys[key] = (tensive, idx)
                backend.write_row(attributes[tensive], idx, backend.from_ndarray(array))
                idx += 1

        return attributes, keys

    # TODO: in principle, should not be needed at all (GPU-resident state)
    def __getitem__(self, item: str):
        idx = self.backend.to_ndarray(self.idx)
        all_valid = idx[:self.SD_num]
        if item == 'n':
            n = self.backend.to_ndarray(self.n)
            result = n[all_valid]
        else:
            tensive = self.keys[item][0]
            attr = self.keys[item][1]
            attribute = self.backend.to_ndarray(self.backend.read_row(self.attributes[tensive], attr))
            result = attribute[all_valid]
        return result

    def get_backend_storage(self, item):
        tensive = self.keys[item][0]
        attr = self.keys[item][1]
        result = self.backend.read_row(self.attributes[tensive], attr)
        return result

    def unsort(self):
        self.backend.shuffle(self.idx, length=self.SD_num, axis=0)

    def min(self, item):
        result = self.backend.amin(self.get_backend_storage(item), self.idx, self.SD_num)
        return result

    def max(self, item):
        result = self.backend.amax(self.get_backend_storage(item), self.idx, self.SD_num)
        return result

    def get_extensive_attrs(self):
        result = self.attributes['extensive']
        return result

    def get_intensive_attrs(self):
        result = self.attributes['intensive']
        return result

    def is_healthy(self):
        return not self.backend.first_element_is_zero(self.healthy)

    # TODO: optionally recycle n=0 drops
    def housekeeping(self):
        if not self.is_healthy():
            self.SD_num = self.backend.remove_zeros(self.n, self.idx, length=self.SD_num)
            self.healthy = self.backend.from_ndarray(np.full((1,), 1))
