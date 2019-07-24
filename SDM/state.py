"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class State:
    def __init__(self, attributes):
        first_attr = None
        for attribute in attributes.values():
            if first_attr is None:
                first_attr = attribute
                continue
            assert attribute.shape == first_attr.shape

        assert first_attr.ndim == 1

        self.keys = {}
        self.data = np.empty((len(attributes), len(first_attr)))

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

    def moment(self, k, m_range=(0, np.inf)):
        idx = np.where(
            np.logical_and(
                self['n'] > 0,  # TODO: alternatively depend on undertaker...
                np.logical_and(m_range[0] <= self['x'], self['x'] < m_range[1])
            )
        )
        if not idx[0].any():
            return 0 if k == 0 else np.nan
        avg, sum = np.average(self['x'][idx] ** k, weights=self['n'][idx], returned=True)
        return avg * sum

    def collide(self, j, k, gamma):
        if self['n'][j] < self['n'][k]:
            j, k = k, j

        gamma = min(gamma, self['n'][j] // self['n'][k])

        if self['n'][k] != 0:  # TODO: guaranteed by undertaker
            n = self['n'][j] - gamma * self['n'][k]
            if n > 0:
                self['n'][j] = n
                self['x'][k] += gamma * self['x'][j]
            else:  # n == 0
                self['n'][j] = self['n'][k] // 2
                self['n'][k] = self['n'][k] - self['n'][j]
                self['x'][j] = gamma * self['x'][j] + self['x'][k]
                self['x'][k] = self['x'][j]

    @property
    def SD_num(self):
        return self.data.shape[1]

    def get_SD(self, i):
        result = self.data[:, i:i+1]
        return result


