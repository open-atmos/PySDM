"""
GPU implementation of pairwise operations backend methods
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class PairMethods(ThrustRTCBackendMethods):
    @cached_property
    def __distance_pair_body(self):
        return trtc.For(
            param_names=("data_out", "data_in", "is_first_in_pair"),
            name_iter="i",
            body="""
        if (is_first_in_pair[i]) {
            data_out[(int64_t)(i/2)] = abs(data_in[i] - data_in[i + 1]);
        }
        """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def distance_pair(self, data_out, data_in, is_first_in_pair, idx):
        perm_in = trtc.DVPermutation(data_in.data, idx.data)
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        self.__distance_pair_body.launch_n(
            len(idx), [data_out.data, perm_in, is_first_in_pair.indicator.data]
        )

    @cached_property
    def __find_pairs_body(self):
        return trtc.For(
            param_names=("cell_start", "perm_cell_id", "is_first_in_pair", "length"),
            name_iter="i",
            body="""
            if (i < length -1) {
                auto is_in_same_cell = perm_cell_id[i] == perm_cell_id[i + 1];
                auto is_even_index = (i - cell_start[perm_cell_id[i]]) % 2 == 0;

                is_first_in_pair[i] = is_in_same_cell && is_even_index;
            } else {
                is_first_in_pair[i] = false;
            }
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    # TODO #330 handle cell_idx (_ below)
    def find_pairs(self, cell_start, is_first_in_pair, cell_id, _, idx):
        perm_cell_id = trtc.DVPermutation(cell_id.data, idx.data)
        d_length = trtc.DVInt64(len(idx))
        self.__find_pairs_body.launch_n(
            n=len(idx),
            args=(
                cell_start.data,
                perm_cell_id,
                is_first_in_pair.indicator.data,
                d_length,
            ),
        )

    @cached_property
    def __max_pair_body(self):
        return trtc.For(
            param_names=("data_out", "perm_in", "is_first_in_pair"),
            name_iter="i",
            body="""
            if (is_first_in_pair[i]) {
                data_out[(int64_t)(i/2)] = max(perm_in[i], perm_in[i + 1]);
            }
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def max_pair(self, data_out, data_in, is_first_in_pair, idx):
        perm_in = trtc.DVPermutation(data_in.data, idx.data)
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        self.__max_pair_body.launch_n(
            len(idx), [data_out.data, perm_in, is_first_in_pair.indicator.data]
        )

    @cached_property
    def __sort_pair_body(self):
        return trtc.For(
            param_names=("data_out", "data_in", "is_first_in_pair"),
            name_iter="i",
            body="""
            if (is_first_in_pair[i]) {
                if (data_in[i] < data_in[i + 1]) {
                    data_out[i] = data_in[i + 1];
                    data_out[i + 1] = data_in[i];
                }
                else {
                    data_out[i] = data_in[i];
                    data_out[i + 1] = data_in[i + 1];
                }
            }
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def sort_pair(self, data_out, data_in, is_first_in_pair, idx):
        perm_in = trtc.DVPermutation(data_in.data, idx.data)
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        if len(idx) > 1:
            self.__sort_pair_body.launch_n(
                len(idx) - 1, [data_out.data, perm_in, is_first_in_pair.indicator.data]
            )

    @cached_property
    def __sort_within_pair_by_attr_body(self):
        return trtc.For(
            param_names=("idx", "is_first_in_pair", "attr"),
            name_iter="i",
            body="""
            if (is_first_in_pair[i]) {
                if (attr[idx[i]] < attr[idx[i + 1]]) {
                    auto tmp = idx[i];
                    idx[i] = idx[i + 1];
                    idx[i + 1] = tmp;
                }
            }
            """,
        )

    def sort_within_pair_by_attr(self, idx, is_first_in_pair, attr):
        if len(idx) < 2:
            return
        self.__sort_within_pair_by_attr_body.launch_n(
            len(idx) - 1, [idx.data, is_first_in_pair.indicator.data, attr.data]
        )

    @cached_property
    def __sum_pair_body(self):
        return trtc.For(
            param_names=("data_out", "perm_in", "is_first_in_pair"),
            name_iter="i",
            body="""
            if (is_first_in_pair[i]) {
                data_out[(int64_t)(i/2)] = perm_in[i] + perm_in[i + 1];
            }
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def sum_pair(self, data_out, data_in, is_first_in_pair, idx):
        perm_in = trtc.DVPermutation(data_in.data, idx.data)
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        self.__sum_pair_body.launch_n(
            n=len(idx),
            args=(data_out.data, perm_in, is_first_in_pair.indicator.data),
        )

    @cached_property
    def __min_pair_body(self):
        return trtc.For(
            param_names=(
                "data_out",
                "data_in",
                "is_first_in_pair",
                "idx",
            ),
            name_iter="i",
            body="""
            if (is_first_in_pair[i]) {
                data_out[(int64_t)(i/2)] = min(data_in[idx[i]], data_in[idx[i + 1]]);
            }
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def min_pair(self, data_out, data_in, is_first_in_pair, idx):
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        self.__min_pair_body.launch_n(
            n=len(idx),
            args=(
                data_out.data,
                data_in.data,
                is_first_in_pair.indicator.data,
                idx.data,
            ),
        )

    @cached_property
    def __multiply_pair_body(self):
        return trtc.For(
            param_names=(
                "data_out",
                "data_in",
                "is_first_in_pair",
                "idx",
            ),
            name_iter="i",
            body="""
            if (is_first_in_pair[i]) {
                data_out[(int64_t)(i/2)] = data_in[idx[i]] * data_in[idx[i + 1]];
            }
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def multiply_pair(self, data_out, data_in, is_first_in_pair, idx):
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        self.__multiply_pair_body.launch_n(
            n=len(idx),
            args=(
                data_out.data,
                data_in.data,
                is_first_in_pair.indicator.data,
                idx.data,
            ),
        )
