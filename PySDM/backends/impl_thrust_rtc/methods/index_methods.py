"""
GPU implementation of shuffling and sorting backend methods
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class IndexMethods(ThrustRTCBackendMethods):
    @cached_property
    def __identity_index_body(self):
        return trtc.For(
            param_names=("idx",),
            name_iter="i",
            body="idx[i] = i;",
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def identity_index(self, idx):
        self.__identity_index_body.launch_n(idx.size(), (idx,))

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def shuffle_global(idx, length, u01):
        # WARNING: ineffective implementation

        # TODO #328 : Thrust modifies key array, conflicts with rand_reuse logic
        # tmpu01 = trtc.device_vector(u01.name_elem_cls(), u01.size())
        # trtc.Copy(u01, tmpu01)
        # trtc.Sort_By_Key(tmpu01.range(0, length), idx.range(0, length))

        trtc.Sort_By_Key(u01.range(0, length), idx.range(0, length))

    @cached_property
    def __shuffle_local_body(self):
        return trtc.For(
            param_names=("cell_start", "u01", "idx"),
            name_iter="i",
            body="""
            for (auto k = cell_start[i+1]-1; k > cell_start[i]; k -= 1) {
                auto j = cell_start[i] + (int64_t)(u01[k] * (cell_start[i+1] - cell_start[i]) );
                auto tmp = idx[k];
                idx[k] = idx[j];
                idx[j] = tmp;
            }
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def shuffle_local(self, idx, u01, cell_start):
        raise AssertionError("Unpredictable behavior")  # TODO #358
        self.__shuffle_local_body.launch_n(  # pylint: disable=unreachable
            cell_start.size() - 1, [cell_start, u01, idx]
        )

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sort_by_key(idx, attr):
        d_attr_data_copy, _, _ = attr._get_empty_data(attr.shape, attr.dtype)
        trtc.Sort_By_Key(d_attr_data_copy, idx.data)
