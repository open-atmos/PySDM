"""
Created at 09.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.backends.numba import Numba

from SDM.backends import pythran_impl
if not hasattr(pythran_impl, '__pythran__'):
    raise ModuleNotFoundError


class Pythran(Numba):
    # @staticmethod
    # def remove_zeros(data, idx, length):
    #     return pythran_impl.remove_zeros(data, idx, length)

    # @staticmethod
    # def coalescence(n, idx, length, intensive, extensive, gamma, healthy):
    #     return pythran_impl.coalescence(n, idx, length, intensive, extensive, gamma, healthy)

    @staticmethod
    def sum_pair(data_out, data_in, idx, length):
        return pythran_impl.sum_pair(data_out, data_in, idx, length)

    @staticmethod
    def max_pair(data_out, data_in, idx, length):
        return pythran_impl.max_pair(data_out, data_in, idx, length)

    @staticmethod
    def sum(data_out, data_in):
        return pythran_impl.sum(data_out, data_in)

    @staticmethod
    def floor(row):
        return pythran_impl.floor(row)
