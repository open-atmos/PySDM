"""
Created at 24.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class Numpy:
    @staticmethod
    def array(shape, type):
        if type is float:
            data = np.full(shape, np.nan, dtype=np.float)
        if type is int:
            data = np.full(shape, -1, dtype=np.int)
        return data

    @staticmethod
    def shuffle(data, axis):
        idx = np.random.permutation(data.shape[axis])
        Numpy.reindex(data, idx)

    @staticmethod
    def reindex(data, idx, axis):
        if axis == 1:
            data = data[:, idx]
        else:
            raise NotImplementedError

    @staticmethod
    def transform(data, func):
        data[:] = np.fromfunction(
            np.vectorize(func, otypes=(data.dtype,)),
            data.shape,
            dtype=np.int
        )

    @staticmethod
    def foreach(data, func):
        for i in range(len(data)):
            func(i)

    @staticmethod
    def shape(data):
        return data.shape

    @staticmethod
    def urand(data, min=0, max=1):
        data[:] = np.random.uniform(min, max, data.shape)




