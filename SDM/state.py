"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from SDM.backends.numpy import Numpy as backend


class State:
    def __init__(self, n: np.ndarray, intensive: dict, extensive: dict, segment_num: int):
        assert n.ndim == 1

        # https://en.wikipedia.org/wiki/Intensive_and_extensive_properties
        for attribute in intensive.values():
            assert backend.shape(attribute) == backend.shape(n)
        for attribute in extensive.values():
            assert backend.shape(attribute) == backend.shape(n)

        self.SD_num = len(n)
        self.idx = backend.from_ndarray(np.arange(self.SD_num))
        self.n = backend.from_ndarray(n)
        self.keys = {}
        self.attributes = {'intensive': {}, 'extensive': {}}
        # TODO clean
        attributes = {'intensive': State.divide_by_type(intensive), 'extensive': State.divide_by_type(extensive)}
        # self.attributes['intensive']['int'] = backend.array((len(attributes['intensive']['int']), self.SD_num), int)
        # self.attributes['intensive']['float'] = backend.array((len(attributes['intensive']['float64']), self.SD_num), float)
        # self.attributes['extensive']['int'] = backend.array((len(attributes['extensive']['int']), self.SD_num), int)
        self.attributes['extensive']['float64'] = backend.array((len(attributes['extensive']['float64']), self.SD_num), float)

        for tensive in self.attributes:
            for dtype in self.attributes[tensive]:
                idx = 0
                for key, array in attributes[tensive][dtype].items():
                    self.keys[key] = (tensive, dtype, idx)
                    self.attributes[tensive][dtype][idx, :] = array[:]
                    idx += 1

        self.segment = backend.from_ndarray(np.full(segment_num, 0))
        self.segment_multiplicity = backend.array((segment_num,), int)
        self.segment_order = backend.array((self.SD_num,), int)

    @staticmethod
    def divide_by_type(attributes: dict) -> dict:
        result = {}
        for key, ndarray in attributes.items():
            dtype = str(ndarray.dtype)
            if dtype not in result:
                result[dtype] = {}
            result[dtype][key] = ndarray
        return result

    # TODO: in principle, should not be needed at all (GPU-resident state)
    def __getitem__(self, item: str) -> backend.storage:
        all_valid = self.idx[0, :self.SD_num]
        if item == 'n':
            result = self.n[0, all_valid]
        else:
            tensive = self.keys[item][0]
            dtype = self.keys[item][1]
            attr = self.keys[item][2]
            result = self.attributes[tensive][dtype][attr, all_valid]
        return result

    @property
    def _n(self):
        return self.n[0]

    @property
    def _idx(self):
        return self.idx[0]

    @property
    def _x(self):
        return self.attributes['extensive']['float64'][0, :]

    def sort_by(self, item: str, stable=False):
        if stable:
            backend.stable_argsort(self.idx, self[item], length=self.SD_num)
        else:
            backend.argsort(self.idx, self[item], length=self.SD_num)

    def unsort(self):
        backend.shuffle(self.idx, length=self.SD_num, axis=1)

    def min(self, item):
        result = backend.amin(self[item])
        return result

    def max(self, item):
        result = backend.amax(self[item])
        return result

    # TODO update
    def moment(self, k, attr='x', attr_range=(0, np.inf)):
        idx = np.where(
            np.logical_and(
                self['n'] > 0,  # TODO: alternatively depend on undertaker...
                np.logical_and(attr_range[0] <= self[attr], self[attr] < attr_range[1])
            )
        )
        if not idx[0].any():
            return 0 if k == 0 else np.nan
        avg, sum = np.average(self[attr][idx] ** k, weights=self['n'][idx], returned=True)
        return avg * sum

    def get_extensive_attrs(self):
        result = self.attributes['extensive']
        return result

    def is_healthy(self):
        result = backend.amin(self.n[0][self.idx[0, 0:self.SD_num]]) > 0
        return result

    def housekeeping(self):
        if self.is_healthy():
            return
        else:
            self.SD_num = backend.remove_zeros(self.n, self.idx, length=self.SD_num)
            print(self.n[0])
            print(self.idx)





