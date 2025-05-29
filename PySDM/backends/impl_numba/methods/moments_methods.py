"""
CPU implementation of moment calculation backend methods
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba.atomic_operations import atomic_add


class MomentsMethods(BackendMethods):
    @cached_property
    def _moments_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(
            *,
            moment_0,
            moments,
            multiplicity,
            attr_data,
            cell_id,
            idx,
            length,
            ranks,
            min_x,
            max_x,
            x_attr,
            weighting_attribute,
            weighting_rank,
            skip_division_by_m0,
        ):
            # pylint: disable=too-many-locals
            moment_0[:] = 0
            moments[:, :] = 0
            for idx_i in numba.prange(length):  # pylint: disable=not-an-iterable
                i = idx[idx_i]
                if min_x <= x_attr[i] < max_x:
                    atomic_add(
                        moment_0,
                        cell_id[i],
                        multiplicity[i] * weighting_attribute[i] ** weighting_rank,
                    )
                    for k in range(ranks.shape[0]):
                        atomic_add(
                            moments,
                            (k, cell_id[i]),
                            (
                                multiplicity[i]
                                * weighting_attribute[i] ** weighting_rank
                                * attr_data[i] ** ranks[k]
                            ),
                        )
            if not skip_division_by_m0:
                for c_id in range(moment_0.shape[0]):
                    for k in range(ranks.shape[0]):
                        moments[k, c_id] = (
                            moments[k, c_id] / moment_0[c_id]
                            if moment_0[c_id] != 0
                            else 0
                        )

        return body

    def moments(
        self,
        *,
        moment_0,
        moments,
        multiplicity,
        attr_data,
        cell_id,
        idx,
        length,
        ranks,
        min_x,
        max_x,
        x_attr,
        weighting_attribute,
        weighting_rank,
        skip_division_by_m0,
    ):
        return self._moments_body(
            moment_0=moment_0.data,
            moments=moments.data,
            multiplicity=multiplicity.data,
            attr_data=attr_data.data,
            cell_id=cell_id.data,
            idx=idx.data,
            length=length,
            ranks=ranks.data,
            min_x=min_x,
            max_x=max_x,
            x_attr=x_attr.data,
            weighting_attribute=weighting_attribute.data,
            weighting_rank=weighting_rank,
            skip_division_by_m0=skip_division_by_m0,
        )

    @cached_property
    def _spectrum_moments_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(
            *,
            moment_0,
            moments,
            multiplicity,
            attr_data,
            cell_id,
            idx,
            length,
            rank,
            x_bins,
            x_attr,
            weighting_attribute,
            weighting_rank,
        ):
            # pylint: disable=too-many-locals
            moment_0[:, :] = 0
            moments[:, :] = 0
            for idx_i in numba.prange(length):  # pylint: disable=not-an-iterable
                i = idx[idx_i]
                for k in range(x_bins.shape[0] - 1):
                    if x_bins[k] <= x_attr[i] < x_bins[k + 1]:
                        atomic_add(
                            moment_0,
                            (k, cell_id[i]),
                            multiplicity[i] * weighting_attribute[i] ** weighting_rank,
                        )
                        atomic_add(
                            moments,
                            (k, cell_id[i]),
                            (
                                multiplicity[i]
                                * weighting_attribute[i] ** weighting_rank
                                * attr_data[i] ** rank
                            ),
                        )
                        break
            for c_id in range(moment_0.shape[1]):
                for k in range(x_bins.shape[0] - 1):
                    moments[k, c_id] = (
                        moments[k, c_id] / moment_0[k, c_id]
                        if moment_0[k, c_id] != 0
                        else 0
                    )

        return body

    def spectrum_moments(
        self,
        *,
        moment_0,
        moments,
        multiplicity,
        attr_data,
        cell_id,
        idx,
        length,
        rank,
        x_bins,
        x_attr,
        weighting_attribute,
        weighting_rank,
    ):
        assert moments.shape[0] == x_bins.shape[0] - 1
        assert moment_0.shape == moments.shape
        return self._spectrum_moments_body(
            moment_0=moment_0.data,
            moments=moments.data,
            multiplicity=multiplicity.data,
            attr_data=attr_data.data,
            cell_id=cell_id.data,
            idx=idx.data,
            length=length,
            rank=rank,
            x_bins=x_bins.data,
            x_attr=x_attr.data,
            weighting_attribute=weighting_attribute.data,
            weighting_rank=weighting_rank,
        )
