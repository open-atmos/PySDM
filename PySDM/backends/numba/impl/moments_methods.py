import numba
from PySDM.backends.numba import conf


class MomentsMethods:
    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def moments_body(
            moment_0, moments, n, attr_data, cell_id, idx, length,
            ranks, min_x, max_x, x_attr, weighting_attribute, weighting_rank):
        moment_0[:] = 0
        moments[:, :] = 0
        for i in idx[:length]:
            if min_x < x_attr[i] < max_x:
                moment_0[cell_id[i]] += n[i] * weighting_attribute[i]**weighting_rank
                for k in range(ranks.shape[0]):  # TODO #401 (AtomicAdd)
                    moments[k, cell_id[i]] += n[i] * weighting_attribute[i]**weighting_rank * attr_data[i] ** ranks[k]
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
