import numba

from PySDM.backends.numba import conf
from PySDM.backends.numba.impl._atomic_operations import atomic_add


class MomentsMethods:
    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def moments_body(
            moment_0, moments, n, attr_data, cell_id, idx, length,
            ranks, min_x, max_x, x_attr, weighting_attribute, weighting_rank):
        moment_0[:] = 0
        moments[:, :] = 0
        for idx_i in numba.prange(length):
            i = idx[idx_i]
            if min_x < x_attr[i] < max_x:
                atomic_add(moment_0, cell_id[i], n[i] * weighting_attribute[i] ** weighting_rank)
                for k in range(ranks.shape[0]):
                    atomic_add(moments, (k, cell_id[i]),
                               n[i] * weighting_attribute[i] ** weighting_rank * attr_data[i] ** ranks[k])
        for c_id in range(moment_0.shape[0]):
            for k in range(ranks.shape[0]):
                moments[k, c_id] = moments[k, c_id] / moment_0[c_id] if moment_0[c_id] != 0 else 0

    @staticmethod
    def moments(
            moment_0, moments, n, attr_data, cell_id, idx, length,
            ranks, min_x, max_x, x_attr, weighting_attribute, weighting_rank):
        return MomentsMethods.moments_body(
            moment_0.data, moments.data, n.data, attr_data.data, cell_id.data,
            idx.data, length, ranks.data,
            min_x, max_x, x_attr.data, weighting_attribute.data, weighting_rank
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def spectrum_moments_body(
            moment_0, moments, n, attr_data, cell_id, idx, length,
            rank, x_bins, x_attr, weighting_attribute, weighting_rank):
        moment_0[:, :] = 0
        moments[:, :] = 0
        for idx_i in numba.prange(length):
            i = idx[idx_i]
            for k in range(x_bins.shape[0] - 1):
                if x_bins[k] <= x_attr[i] < x_bins[k + 1]:
                    atomic_add(moment_0, (k, cell_id[i]),
                               n[i] * weighting_attribute[i] ** weighting_rank)
                    atomic_add(moments, (k, cell_id[i]),
                               n[i] * weighting_attribute[i] ** weighting_rank * attr_data[i] ** rank)
                    break
        for c_id in range(moment_0.shape[1]):
            for k in range(x_bins.shape[0] - 1):
                moments[k, c_id] = moments[k, c_id] / moment_0[k, c_id] if moment_0[k, c_id] != 0 else 0

    @staticmethod
    def spectrum_moments(
            moment_0, moments, n, attr_data, cell_id, idx, length,
            rank, x_bins, x_attr, weighting_attribute, weighting_rank):
        assert moments.shape[0] == x_bins.shape[0] - 1
        assert moment_0.shape == moments.shape
        return MomentsMethods.spectrum_moments_body(
            moment_0.data, moments.data, n.data, attr_data.data, cell_id.data,
            idx.data, length, rank,
            x_bins.data, x_attr.data, weighting_attribute.data, weighting_rank
        )
