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
            for idx_i in range(length):  # pylint: disable=not-an-iterable
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
    # @jax.jit
    def _spectrum_moments_body(self):
        # @numba.njit(**self.default_jit_flags)
        # @partial(jax.jit, static_argnums=(6,))
        @jax.jit
        def body(
            # *,
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
            bin_to_count,
            idx_i
            # indices
            # truth_table,
        ):
            # moment_0 - 1
            # moments - 1
            # x_attr - 0
            # multiplicity - 0
            # weighting_attribute - 0
            # attr_data - 0

            i = idx[idx_i]

# (k[0] > 0 & ~(x_bins[k[0]-1] <= x_attr < x_bins[k[0]])) | 
            moment_0 = moment_0.at[bin_to_count, cell_id[i]].add(multiplicity[i] * weighting_attribute[i] ** weighting_rank)
            moments = moments.at[bin_to_count, cell_id[i]].add(multiplicity[i] * weighting_attribute[i] ** weighting_rank * attr_data[i] ** rank)


            # for k in range(x_bins.shape[0] - 1):
            #     if (x_bins[k] <= x_attr) & (x_attr < x_bins[k+1]):
            #         k > 0 and not (x_bins[k-1] <= x_attr < x_bins[k]) or k==0
            #         moment_0 = moment_0.at[k].add(jax.numpy.multiply(multiplicity, weighting_attribute) ** weighting_rank)
            #         moments = moments.at[k].add(jax.numpy.multiply(jax.numpy.multiply(multiplicity, weighting_attribute) ** weighting_rank, attr_data ** rank))
            #         break

            # Thing 2 (moments = this thing in another func or sth, if we even want to parallelize it)
            # np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            
            return moment_0, moments
            # # This part below is only needed to scale it down, so focus on the first part working first

            # for k in range(x_bins.shape[0] - 1):
            #     moments = moments.at[k,:].divide(moment_0[k,:])
            #     # moments = jax.numpy.divide(moments, moment_0)
    

        return body

    # @jax.jit
    # def generate_calculate_indices():
    #     for k in range(x_bins.shape[0] - 1):
    #         if (x_bins[k] <= x_attr) & (x_attr < x_bins[k+1]):
    #             moments
    #     pass
    # @jax.jit
    

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
        # truth_table = []
        
        # for k in range(x_bins.shape[0] - 1):
        #     truth_table.append((x_bins[k] <= x_attr) & (x_attr < x_bins[k+1]))
                
        
        # indices = jax.numpy.argwhere(truth_table)

        # print("done")
        @jax.jit
        def spectrum_moments_helper(x_bins, x_attr, idx, idx_i):
            def cond_fun(k):
                return ((x_bins[k] > x_attr[i]) | (x_attr[i] > x_bins[k+1])) & (k < x_bins.shape[0] - 1)
            i = idx[idx_i]
            bin_to_calculate = jax.lax.while_loop(cond_fun, lambda k: k+1, 0)
            return bin_to_calculate

        moment_0.data = moment_0.data.at[:, :].set(0)
        moments.data = moments.data.at[:,:].set(0)
        idx_idxs = jax.numpy.arange(length-1)

        # maybe vmap just the indexes???
        count_bins_func = jax.vmap(spectrum_moments_helper, (None, None, None, 0))
        bins_to_count = count_bins_func(x_bins.data, x_attr.data, idx.data, idx_idxs)
        mapped_spectrum = jax.vmap(self._spectrum_moments_body, (None, None, None, None, None, None, None, None, None, None, None, None, 0, 0))

        moment_0.data, moments.data = mapped_spectrum(
            moment_0.data,
            moments.data,            
            multiplicity.data,
            attr_data.data,
            cell_id.data,
            idx.data,
            length,
            rank,
            x_bins.data,
            x_attr.data,
            weighting_attribute.data,
            weighting_rank,
            bins_to_count,
            idx_idxs
        )

        # This sum takes too much time ???
        moments.data = moments.data.sum(0)
        moment_0.data = moment_0.data.sum(0)
        # moment_0.data, moments.data = 
        # self._spectrum_moments_body(
        #     moment_0.data,
        #     moments.data,
        #     multiplicity.data,
        #     attr_data.data,
        #     cell_id.data,
        #     idx.data,
        #     length,
        #     rank,
        #     x_bins.data,
        #     x_attr.data,
        #     weighting_attribute.data,
        #     weighting_rank,
        #     # indices,
        #     # truth_table=truth_table,
        # )

        # print(moments.data)
        # return self._spectrum_moments_body(
        #     moment_0=moment_0.data,
        #     moments=moments.data,
        #     multiplicity=multiplicity.data,
        #     attr_data=attr_data.data,
        #     cell_id=cell_id.data,
        #     idx=idx.data,
        #     length=length,
        #     rank=rank,
        #     x_bins=x_bins.data,
        #     x_attr=x_attr.data,
        #     weighting_attribute=weighting_attribute.data,
        #     weighting_rank=weighting_rank,
        # )
