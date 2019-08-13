"""
Created at 09.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.backends.numba import Numba

# import os
# os.environ['OMP_NUM_THREADS'] = '8'
#%%
from SDM.backends import pythran_impl
if not hasattr(pythran_impl, '__pythran__'):
    # TODO: use fluentPythran
    import subprocess, importlib, os
    os.chdir("SDM/backends")
    code = subprocess.call(["pythran", "pythran_impl.py"])
    os.chdir("../..")
    assert code == 0
    importlib.reload(pythran_impl)
#%%

class Pythran(Numba):
    @staticmethod
    def remove_zeros(data, idx, length):
        return pythran_impl.remove_zeros(data, idx, length)

    @staticmethod
    def coalescence(n, idx, length, intensive, extensive, gamma, healthy):
        return pythran_impl.coalescence(n, idx, length, intensive, extensive, gamma, healthy)

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

    @staticmethod
    def amin(row, idx, length):
        return pythran_impl.amin(row, idx, length)

    @staticmethod
    def amax(row, idx, length):
        return pythran_impl.amax(row, idx, length)

    @staticmethod
    def multiply(data, multiplier):
        return pythran_impl.multiply(data, multiplier)

    @staticmethod
    def sum(data_out, data_in):
        return pythran_impl.sum(data_out, data_in)

    @staticmethod
    def first_element_is_zero(arr):
        return pythran_impl.first_element_is_zero(arr)

