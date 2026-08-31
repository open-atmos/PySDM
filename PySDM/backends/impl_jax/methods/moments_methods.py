"""
JAX implementation of moment calculation backend methods
"""

from functools import cached_property

import jax
import jax.numpy as jnp

from PySDM.backends.impl_common.backend_methods import BackendMethods


@jax.jit
def moments_helper(min_x, max_x, x_attr, idx, idx_i):
    i = idx[idx_i]
    return (min_x <= x_attr[i]) & (x_attr[i] < max_x)


@jax.jit
def spectrum_moments_helper(x_bins, x_attr, idx, idx_i):
    def cond_fun(k):
        return (k < x_bins.shape[0] - 1) & (
            (x_bins[k] > x_attr[i]) | (x_attr[i] > x_bins[k + 1])
        )

    i = idx[idx_i]
    return jax.lax.while_loop(cond_fun, lambda k: k + 1, 0)


class MomentsMethods(BackendMethods):
    @cached_property
    def _moments_body(self):
        @jax.jit
        def body(  # pylint: disable=too-many-positional-arguments
            moment_0,
            moments,
            multiplicity,
            attr_data,
            x_attr,
            min_x,
            max_x,
            ranks,
            weighting_attribute,
            weighting_rank,
        ):
            assert len(ranks) == 1
            k = range(ranks.shape[0])
            count_element_flag = (min_x <= x_attr) & (x_attr < max_x)
            moment_0 += count_element_flag * multiplicity * weighting_attribute ** weighting_rank

            moments = moments.at[k].add(
                count_element_flag * multiplicity * weighting_attribute ** weighting_rank * attr_data ** ranks[k]
            )

            return moment_0, moments

        return body

    def moments(  # pylint: disable=too-many-locals
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
        # This method isnt used in shima example
        mapped_moments = jax.vmap(
            self._moments_body,
            in_axes=(0, 1, 0, 0, 0, None, None, None, 0, None),
            out_axes=(0, 1)
        )
        moments.data = moments.data.at[:, :].set(0)
        moment_0.data = moment_0.data.at[:].set(0)

        moment_0.data, moments.data = mapped_moments(
            moment_0.data[cell_id.data],
            moments.data[cell_id.data],
            multiplicity.data[idx.data],
            attr_data.data[idx.data],
            x_attr.data[idx.data],
            min_x,
            max_x,
            ranks.data,
            weighting_attribute.data[idx.data],
            weighting_rank,
        )

        moment_0.data.block_until_ready()
        moments.data = jnp.sum(moments.data, axis=1, keepdims=True) # This won't work for multi-cell
        # moment_0.data = jnp.sum(moment_0.data, axis=0, keepdims=True) # This won't work for multi-cell



        if not skip_division_by_m0:
            moments.data = jnp.where(
                moment_0.data != 0, moments.data / moment_0.data, 0.0
            )

    @cached_property
    def _spectrum_moments_body(self):
        @jax.jit
        def body(  # pylint: disable=too-many-positional-arguments
            moment_0,
            moments,
            multiplicity,
            attr_data,
            x_attr,
            x_bins,
            # cell_id,
            # idx,
            rank,
            weighting_attribute,
            weighting_rank,
            # bin_to_count,
            # idx_i,
        ):
            def loop_break_cond(cond_args):
                k, loop_break, _, _, x_bins = cond_args

                return ~((k == x_bins.shape[0] - 1) | loop_break)

            # i = idx[idx_i]
            def loop_body(loop_args):
                k, loop_break, moment_0, moments, x_bins = loop_args

                loop_break = (x_bins[k] <= x_attr) & (x_attr < x_bins[k + 1])
                moment_0 = moment_0.at[k].add(loop_break * multiplicity * weighting_attribute ** weighting_rank)
                moments = moments.at[k].add(loop_break * multiplicity * weighting_attribute ** weighting_rank * attr_data ** rank)
                return (k+1, loop_break, moment_0, moments, x_bins)
            _, _, moment_0, moments, _ = jax.lax.while_loop(loop_break_cond, loop_body, (0, False, moment_0, moments, x_bins))
            # moment_0 = moment_0.at[bin_to_count, cell_id[i]].add(
            #     multiplicity[i] * weighting_attribute[i] ** weighting_rank
            # )
            # moments = moments.at[bin_to_count, cell_id[i]].add(
            #     multiplicity[i]
            #     * weighting_attribute[i] ** weighting_rank
            #     * attr_data[i] ** rank
            # )

            return moment_0, moments

        return body

    def spectrum_moments(  # pylint: disable=too-many-locals
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
        moments.data = moments.data.at[:, :].set(0)
        moment_0.data = moment_0.data.at[:, :].set(0)
        idx_i = jnp.arange(length)

        mapped_spectrum_moments = jax.vmap(self._spectrum_moments_body,
                                           in_axes=(1, 1, 0, 0, 0, None, None, 0, None),
                                           out_axes=(1,1))

        moment_0.data, moments.data = mapped_spectrum_moments(
            moment_0.data[:, cell_id.data[idx.data[idx_i]]],
            moments.data[:, cell_id.data[idx.data[idx_i]]],
            multiplicity.data[idx.data[idx_i]],
            attr_data.data[idx.data[idx_i]],
            x_attr.data[idx.data[idx_i]],
            x_bins.data,
            rank,
            weighting_attribute.data[idx.data[idx_i]],
            weighting_rank,
        )
        moment_0.data.block_until_ready()
        moments.data = jnp.sum(moments.data, axis=1, keepdims=True) # This won't work for multi-cell
        moment_0.data = jnp.sum(moment_0.data, axis=1, keepdims=True) # This won't work for multi-cell
 

        if not skip_division_by_m0:
            moments.data = jnp.where(
                moment_0.data != 0, moments.data / moment_0.data, 0.0
            )
        
