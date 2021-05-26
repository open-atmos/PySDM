import numba
from PySDM.backends.numba import conf


class IndexMethods:

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def identity_index(idx):
        for i in numba.prange(len(idx)):
            idx[i] = i

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def shuffle_global(idx, length, u01):
        for i in range(length-1, 0, -1):
            j = int(u01[i] * (i+1))
            idx[i], idx[j] = idx[j], idx[i]

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def shuffle_local(idx, u01, cell_start):
        for c in numba.prange(len(cell_start) - 1):
            for i in range(cell_start[c+1]-1, cell_start[c], -1):
                j = int(cell_start[c] + u01[i] * (cell_start[c+1] - cell_start[c]))
                idx[i], idx[j] = idx[j], idx[i]

    @staticmethod
    def sort_by_key(idx, attr):
        idx.data[:] = attr.data.argsort(kind="stable")[::-1]
