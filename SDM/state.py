"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class State:
    def __init__(self, m, n):
        assert m.shape == n.shape
        assert len(m.shape) == 1

        self.data = np.concatenate((m.copy()[:, None], n.copy()[:, None]), axis=1)

    @property
    def x(self):
        return self.data[:, 0]

    @x.setter
    def x(self, value):
        self.data[:, 0] = value

    @property
    def n(self):
        return self.data[:, 1]

    @n.setter
    def n(self, value):
        self.data[:, 1] = value

    def __sort(self, key):
        idx = np.argsort(key)
        self.n = self.n[idx]
        self.x = self.x[idx]

    def sort_by_m(self):
        self.__sort(self.x)

    def sort_by_n(self):
        self.__sort(self.n)

    def unsort(self):
        idx = np.random.permutation(np.arange(len(self)))
        self.x = self.x[idx]
        self.n = self.n[idx]

    def m_min(self):
        result = np.amin(self.x)
        return result

    def m_max(self):
        result = np.amax(self.x)
        return result

    def moment(self, k, m_range=(0, np.inf)):
        idx = np.where(
            np.logical_and(
                self.n > 0,  # TODO: alternatively depend on undertaker...
                np.logical_and(m_range[0] <= self.x, self.x < m_range[1])
            )
        )
        if not idx[0].any():
            return 0 if k == 0 else np.nan
        avg, sum = np.average(self.x[idx] ** k, weights=self.n[idx], returned=True)
        return avg * sum

    def collide(self, i, j, gamma):
        if self.n[i] < self.n[j]:
            i, j = j, i

        gamma = min(gamma, self.n[i] // self.n[j])

        if self.n[j] != 0:
            self.n[i] -= gamma * self.n[j]
            self.x[j] += gamma * self.x[i]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.n[item]
