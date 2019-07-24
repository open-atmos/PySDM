"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class State:
    def __init__(self, attributes):
#        assert n.ndim == 1
#        for attribute in attributes.values():
#            assert attribute.shape == n.shape

#        self.n = n
        n = attributes["n"]
        self.keys = {}
        self.data = np.empty((len(attributes), len(n)))

        idx = 0
        for key, array in attributes.items():
            self.keys[key] = idx
            self.data[idx, :] = array[:]
            idx += 1

    def __getitem__(self, item):
        result = self.data[self.keys[item], :]
        return result

    def _reindex(self, idx):
        self.data[:] = self.data[:, idx]

    def sort_by(self, item):
        idx = self.data[item].argsort()
        self._reindex(idx)

    def unsort(self):
        idx = np.random.permutation(range(self.SD_num))
        self._reindex(idx)

    def min(self, item):
        result = np.amin(self[item])
        return result

    def max(self, item):
        result = np.amax(self[item])
        return result

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

    @property
    def SD_num(self):
        return self.data.shape[1]

    def get_SD(self, i):
        result = self.data[:, i:i+1]
        return result


