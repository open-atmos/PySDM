"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from SDM.backends.default import Default


class State:
    def __init__(self, n: np.ndarray, intensive: dict, extensive: dict, segment_num: int, backend=Default):
        assert State.check_args(n, intensive, extensive)

        self.backend = backend

        self.SD_num = len(n)
        self.idx = backend.from_ndarray(np.arange(self.SD_num))
        self.n = backend.from_ndarray(n)
        self.attributes, self.keys = State.init_attributes_and_keys(self.backend, intensive, extensive, self.SD_num)

        self.segment = backend.from_ndarray(np.full(segment_num, 0))
        self.segment_multiplicity = backend.array((segment_num,), int)
        self.segment_order = backend.array((self.SD_num,), int)

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
                backend.write_row(attributes[tensive], idx, array)
                idx += 1

        return attributes, keys

    # TODO: in principle, should not be needed at all (GPU-resident state)
    def __getitem__(self, item: str):
        all_valid = self.idx[:self.SD_num]
        if item == 'n':
            result = self.n[all_valid]
        else:
            tensive = self.keys[item][0]
            attr = self.keys[item][1]
            result = self.attributes[tensive][attr, all_valid]
        return result

    def get_backend_storage(self, item):
        tensive = self.keys[item][0]
        attr = self.keys[item][1]
        result = self.backend.read_row(self.attributes[tensive], attr)
        return result

    # def sort_by(self, item: str, stable=False):
    #     if stable:
    #         self.backend.stable_argsort(self.idx, self[item], length=self.SD_num)
    #     else:
    #         self.backend.argsort(self.idx, self[item], length=self.SD_num)

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

    def is_healthy(self):
        result = self.backend.amin(self.n, self.idx, self.SD_num) > 0
        return result

    # TODO: optionally recycle n=0 drops
    def housekeeping(self):
        if self.is_healthy():
            return
        else:
            self.SD_num = self.backend.remove_zeros(self.n, self.idx, length=self.SD_num)
