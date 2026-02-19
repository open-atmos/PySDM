"""
CPU implementation of moment calculation backend methods
"""

from functools import cached_property, partial
import time
import jax

# import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba.atomic_operations import atomic_add


class MomentsMethods(BackendMethods):
    @cached_property
    def _moments_body(self):
        # @numba.njit(**self.default_jit_flags)
        @jax.jit
        def body(
            moment_0,
            moments,
            multiplicity,
            attr_data,
            cell_id,
            idx,
            length,
            ranks,
            x_attr,
            weighting_attribute,
            weighting_rank,
            count_element_flags,
            idx_i,
            skip_division_by_m0,
        ):
            assert len(ranks) == 1
            k = 0
            # pylint: disable=too-many-locals
            i = idx[idx_i]

            moment_0 = moment_0.at[cell_id[i]].add(count_element_flags[i] * multiplicity[i] * weighting_attribute[i] ** weighting_rank)
            moments = moments.at[k, cell_id[i]].add(count_element_flags[i] * multiplicity[i] * weighting_attribute[i] ** weighting_rank * attr_data[i] ** ranks[k])


            return moment_0, moments

            # if not skip_division_by_m0:
            #     for c_id in range(moment_0.shape[0]):
            #         for k in range(ranks.shape[0]):
            #             moments[k, c_id] = (
            #                 moments[k, c_id] / moment_0[c_id]
            #                 if moment_0[c_id] != 0
            #                 else 0
            #             )

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
        @jax.jit
        def moments_helper(min_x, max_x, x_attr, idx, idx_i):
            i = idx[idx_i]
            return (min_x <= x_attr[i]) & (x_attr[i] < max_x)

        moment_0.data = moment_0.data.at[:].set(0)
        moments.data = moments.data.at[:,:].set(0)
        idx_idxs = jax.numpy.arange(length)

        count_bins_func = jax.vmap(moments_helper, (None, None, None, None, 0))
        idx_to_count = count_bins_func(min_x, max_x, x_attr.data, idx.data, idx_idxs)
        mapped_spectrum = jax.vmap(self._moments_body, (None, None, None, None, None, None, None, None, None, None, None, None, 0, None))

        moment_0.data, moments.data = mapped_spectrum(
            moment_0.data,
            moments.data,            
            multiplicity.data,
            attr_data.data,
            cell_id.data,
            idx.data,
            length,
            ranks.data,
            x_attr.data,
            weighting_attribute.data,
            weighting_rank,
            idx_to_count,
            idx_idxs,
            skip_division_by_m0
        )

        moments.data = moments.data.sum(0)
        moment_0.data = moment_0.data.sum(0)

        if not skip_division_by_m0:
            moments.data = jax.numpy.where(moment_0.data != 0, moments.data / moment_0.data, 0.0)

    @cached_property
    def _spectrum_moments_body(self):
        @jax.jit
        def body(
            # *,
            moment_0,
            moments,
            multiplicity,
            attr_data,
            cell_id,
            idx,
            rank,
            weighting_attribute,
            weighting_rank,
            bin_to_count,
            idx_i
        ):

            i = idx[idx_i]
            moment_0 = moment_0.at[bin_to_count, cell_id[i]].add(multiplicity[i] * weighting_attribute[i] ** weighting_rank)
            moments = moments.at[bin_to_count, cell_id[i]].add(multiplicity[i] * weighting_attribute[i] ** weighting_rank * attr_data[i] ** rank)

            return moment_0, moments
    

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
        skip_division_by_m0,
    ):
        assert moments.shape[0] == x_bins.shape[0] - 1
        assert moment_0.shape == moments.shape

        @jax.jit
        def spectrum_moments_helper(x_bins, x_attr, idx, idx_i):
            def cond_fun(k):
                return (k < x_bins.shape[0] - 1) & ((x_bins[k] > x_attr[i]) | (x_attr[i] > x_bins[k+1]))
            i = idx[idx_i]
            bin_to_calculate = jax.lax.while_loop(cond_fun, lambda k: k+1, 0)
            return bin_to_calculate

        new_moment_0 = jax.numpy.zeros((moment_0.shape[0]+1, moment_0.shape[1]))
        new_moments = jax.numpy.zeros((moment_0.shape[0]+1, moment_0.shape[1]))
        idx_idxs = jax.numpy.arange(length)

        count_bins_func = jax.vmap(spectrum_moments_helper, (None, None, None, 0))
        bins_to_count = count_bins_func(x_bins.data, x_attr.data, idx.data, idx_idxs)
        assert all(bins_to_count < new_moments.shape[0])
        mapped_spectrum = jax.vmap(self._spectrum_moments_body, (None, None, None, None, None, None, None, None, None, 0, 0))

        new_moment_0, new_moments = mapped_spectrum(
            new_moment_0,
            new_moments,            
            multiplicity.data,
            attr_data.data,
            cell_id.data,
            idx.data,
            rank,
            weighting_attribute.data,
            weighting_rank,
            bins_to_count,
            idx_idxs
        )

        moments.data =  jax.numpy.sum(new_moments[:, :-1, :], axis=0)
        moment_0.data =  jax.numpy.sum(new_moment_0[:, :-1, :], axis=0)

        if not skip_division_by_m0:
            moments.data = jax.numpy.where(moment_0.data != 0, moments.data / moment_0.data, 0.0)
