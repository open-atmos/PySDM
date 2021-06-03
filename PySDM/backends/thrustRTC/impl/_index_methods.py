from .precision_resolver import PrecisionResolver
from ..conf import trtc
import numpy as np
from PySDM.backends.thrustRTC.impl.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS


class IndexMethods:

    __identity_index_body = trtc.For(['idx'], 'i', '''
        idx[i] = i;
    ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def identity_index(idx):
        IndexMethods.__identity_index_body.launch_n(idx.size(), (idx,))

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def shuffle_global(idx, length, u01):
        # WARNING: ineffective implementation

        # TODO #328 : Thrust modifies key array, conflicts with rand_reuse logic
        # tmpu01 = trtc.device_vector(u01.name_elem_cls(), u01.size())
        # trtc.Copy(u01, tmpu01)
        # trtc.Sort_By_Key(tmpu01.range(0, length), idx.range(0, length))

        trtc.Sort_By_Key(u01.range(0, length), idx.range(0, length))

    __shuffle_local_body = trtc.For(['cell_start', 'u01', 'idx'], "i", '''
        for (auto k = cell_start[i+1]-1; k > cell_start[i]; k -= 1) {
            auto j = cell_start[i] + (int64_t)(u01[k] * (cell_start[i+1] - cell_start[i]) );
            auto tmp = idx[k];
            idx[k] = idx[j];
            idx[j] = tmp;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def shuffle_local(idx, u01, cell_start):
        raise NotImplementedError("Unpredictable behavior")  # TODO #358
        IndexMethods.__shuffle_local_body.launch_n(cell_start.size() - 1, [cell_start, u01, idx])

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sort_by_key(idx, attr):
        d_attr_data_copy, _, _ = attr._get_empty_data(attr.shape, attr.dtype)
        trtc.Sort_By_Key(d_attr_data_copy, idx.data)
